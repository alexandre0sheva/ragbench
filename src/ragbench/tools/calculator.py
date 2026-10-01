"""`calculator`: arithmetic through a whitelist AST walk. There is no `eval`: anything outside the whitelist is rejected, and the size of results is capped."""

from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from ragbench.registry import TOOLS
from ragbench.tools.base import BaseTool, ToolContext, ToolResult, make_spec

MAX_EXPRESSION_CHARS = 500
MAX_NODES = 200
MAX_EXPONENT = 1000  # larger exponents are refused unless the base is 0, 1 or -1
MAX_RESULT_BITS = 10_000  # an integer result may have at most this many bits (about 3000 digits)
CONSTANTS = {"pi": math.pi, "e": math.e}


class CalculatorError(ValueError):
    pass


Number = int | float
_BINARY: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _check(value: Any) -> Number:
    if isinstance(value, float) and not math.isfinite(value):
        raise CalculatorError("the result is not a finite number")
    if isinstance(value, int) and value.bit_length() > MAX_RESULT_BITS:
        raise CalculatorError("the result is too large")
    return value


def _power(base: Number, exponent: Number) -> Number:
    if base not in (0, 1, -1) and abs(exponent) > MAX_EXPONENT:
        raise CalculatorError(f"the exponent is too large (limit {MAX_EXPONENT})")
    if isinstance(base, int) and isinstance(exponent, int) and abs(base) > 1 and exponent > 0 and base.bit_length() * exponent > MAX_RESULT_BITS:
        raise CalculatorError("the result is too large")
    try:
        result = base**exponent
    except ZeroDivisionError:
        raise CalculatorError("division by zero") from None
    except OverflowError:
        raise CalculatorError("the result is too large") from None
    if isinstance(result, complex):
        raise CalculatorError("the result is not a real number")
    return _check(result)


def _numbers(arguments: list[Any]) -> list[Number]:
    """Function arguments: plain numbers, or one list / tuple of numbers (`sum([1, 2])`)."""
    flat: list[Number] = []
    for argument in arguments:
        items = argument if isinstance(argument, list | tuple) else [argument]
        if not all(_is_number(item) for item in items):
            raise CalculatorError("functions take numbers")
        flat.extend(items)
    if not flat:
        raise CalculatorError("the function needs at least one number")
    return flat


def _round(arguments: list[Any]) -> Number:
    if len(arguments) not in (1, 2) or not all(_is_number(a) for a in arguments):
        raise CalculatorError("round takes a number and optionally the digits to keep")
    digits = int(arguments[1]) if len(arguments) == 2 else 0
    if not -15 <= digits <= 15:
        raise CalculatorError("round keeps between -15 and 15 digits")
    return round(arguments[0], digits) if len(arguments) == 2 else int(round(arguments[0]))


FUNCTIONS: dict[str, Callable[[list[Any]], Number]] = {
    "round": _round,
    "min": lambda arguments: min(_numbers(arguments)),
    "max": lambda arguments: max(_numbers(arguments)),
    "sum": lambda arguments: _check(sum(_numbers(arguments))),
    "abs": lambda arguments: abs(_numbers(arguments)[0]) if len(arguments) == 1 else _raise("abs takes one number"),
}


def _raise(message: str) -> Number:
    raise CalculatorError(message)


def _evaluate(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        if not _is_number(node.value):
            raise CalculatorError("only numbers are allowed")
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in CONSTANTS:
            raise CalculatorError(f"unknown name {node.id!r}")
        return CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        operand = _evaluate(node.operand)
        if not _is_number(operand):
            raise CalculatorError("unary signs apply to numbers")
        return -operand if isinstance(node.op, ast.USub) else operand
    if isinstance(node, ast.BinOp):
        left, right = _evaluate(node.left), _evaluate(node.right)
        if not (_is_number(left) and _is_number(right)):
            raise CalculatorError("operators apply to numbers")
        if isinstance(node.op, ast.Pow):
            return _power(left, right)
        function = _BINARY.get(type(node.op))
        if function is None:
            raise CalculatorError(f"the operator {type(node.op).__name__} is not allowed")
        try:
            return _check(function(left, right))
        except ZeroDivisionError:
            raise CalculatorError("division by zero") from None
        except OverflowError:
            raise CalculatorError("the result is too large") from None
    if isinstance(node, ast.List | ast.Tuple):
        return [_evaluate(item) for item in node.elts]
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS or node.keywords:
            name = repr(node.func.id) if isinstance(node.func, ast.Name) else "that call"
            raise CalculatorError(f"{name} is not an allowed function (allowed: {', '.join(FUNCTIONS)})")
        return _check(FUNCTIONS[node.func.id]([_evaluate(argument) for argument in node.args]))
    raise CalculatorError(f"{type(node).__name__} is not allowed; use numbers, + - * / // % **, parentheses and {', '.join(FUNCTIONS)}")


def normalize(expression: str) -> str:
    """The notations models write that Python would not read: `^` for power, `×` / `÷`, `$`, thousands separators, a trailing `=`."""
    text = expression.strip().rstrip("=?").strip()
    text = text.replace("×", "*").replace("÷", "/").replace("^", "**").replace("$", "").replace("−", "-")
    return re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)


def calculate(expression: str) -> Number:
    """Evaluate an arithmetic expression. Raises `CalculatorError` for anything that is not plain arithmetic."""
    text = normalize(expression)
    if not text:
        raise CalculatorError("the expression is empty")
    if len(text) > MAX_EXPRESSION_CHARS:
        raise CalculatorError(f"the expression is too long (limit {MAX_EXPRESSION_CHARS} characters)")
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        raise CalculatorError("could not read the expression; write plain arithmetic such as (1200 - 950) / 950 * 100") from None
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise CalculatorError("the expression is too complicated")
    try:
        result = _evaluate(tree)
    except RecursionError:
        raise CalculatorError("the expression is nested too deeply") from None
    if not _is_number(result):
        raise CalculatorError("the expression does not evaluate to one number")
    return result


def format_number(value: Number) -> str:
    if isinstance(value, int):
        return str(value)
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return format(value, ".12g")


class CalculatorArgs(BaseModel):
    expression: str = Field(description="Arithmetic to evaluate, e.g. `(1200 - 950) / 950 * 100`. Supports + - * / // % ** (or ^), parentheses, round(x, digits), min, max, sum, abs, pi and e.")


@TOOLS.register("calculator")
class CalculatorTool(BaseTool):
    Args = CalculatorArgs
    spec = make_spec("calculator", "Evaluate an arithmetic expression exactly. Use it for any calculation instead of doing the arithmetic yourself.", CalculatorArgs)

    def _run(self, args: CalculatorArgs, ctx: ToolContext) -> ToolResult:
        try:
            value = calculate(args.expression)
        except CalculatorError as exc:
            return ToolResult.fail(str(exc))
        return ToolResult.ok(f"{args.expression.strip()} = {format_number(value)}", data=value)
