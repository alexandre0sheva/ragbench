"""User-written tools: any Python callable with type hints, referenced as `{name: my_tool, path: "my_pkg.tools:lookup_price"}`."""

from __future__ import annotations

import importlib
import inspect
import json
import re
import typing
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from ragbench.tools.base import SideEffects, ToolContext, ToolResult, ToolSpec, describe_validation_error, make_spec

NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
SIDE_EFFECTS = ("none", "network", "filesystem")
CUSTOM_OPTIONS = {"name", "path", "description", "side_effects", "timeout_s"}


class CustomToolError(ValueError):
    """A custom tool could not be loaded; the message says what to fix."""


def import_callable(path: str) -> Callable[..., Any]:
    module_name, separator, attribute = path.partition(":")
    if not separator or not module_name or not attribute:
        raise CustomToolError(f"path {path!r} must look like 'package.module:function'")
    try:
        target: Any = importlib.import_module(module_name)
    except ImportError as exc:
        raise CustomToolError(f"cannot import {module_name!r} for the tool at {path!r}: {exc}") from None
    for part in attribute.split("."):
        try:
            target = getattr(target, part)
        except AttributeError:
            raise CustomToolError(f"{module_name!r} has no attribute {attribute!r}") from None
    if not callable(target):
        raise CustomToolError(f"{path!r} is not callable")
    return target


class CallableTool:
    """Wraps a typed Python function as a `Tool`. Its parameters become the JSON schema (via a generated Pydantic model),
    its docstring the description. A parameter named `ctx` (or annotated `ToolContext`) receives the `ToolContext` and is not shown to the model."""

    def __init__(self, name: str, function: Callable[..., Any], *, description: str | None = None, side_effects: SideEffects = "none", timeout_s: float | None = None):
        if not NAME_PATTERN.match(name):
            raise CustomToolError(f"tool name {name!r} must be 1-64 letters, digits, '_' or '-'")
        self.function = function
        self.side_effects: SideEffects = side_effects
        self.timeout_s = timeout_s
        self._ctx_parameter: str | None = None
        self.Args = self._args_model(name, function)
        doc = description or (inspect.getdoc(function) or "").split("\n\n")[0].strip()
        self.spec: ToolSpec = make_spec(name, " ".join(doc.split()) or f"Custom tool {name}.", self.Args)

    def _args_model(self, name: str, function: Callable[..., Any]) -> type[BaseModel]:
        try:
            hints = typing.get_type_hints(function, include_extras=True)
        except Exception as exc:
            raise CustomToolError(f"tool {name!r}: cannot read the type hints of {function!r}: {exc}") from None
        fields: dict[str, Any] = {}
        for parameter in inspect.signature(function).parameters.values():
            if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                raise CustomToolError(f"tool {name!r}: *args / **kwargs parameters are not supported; name each parameter")
            annotation = hints.get(parameter.name)
            if parameter.name == "ctx" or annotation is ToolContext:
                self._ctx_parameter = parameter.name
                continue
            if annotation is None:
                raise CustomToolError(f"tool {name!r}: parameter {parameter.name!r} needs a type hint")
            fields[parameter.name] = (annotation, ... if parameter.default is parameter.empty else parameter.default)
        return create_model(f"{name}_args", __config__=ConfigDict(extra="forbid"), **fields)

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            parsed = self.Args.model_validate(args)
        except ValidationError as exc:
            return ToolResult.fail(describe_validation_error(exc))
        call_args = {field: getattr(parsed, field) for field in type(parsed).model_fields}
        if self._ctx_parameter:
            call_args[self._ctx_parameter] = ctx
        return _as_result(self.function(**call_args))


def _as_result(value: Any) -> ToolResult:
    if isinstance(value, ToolResult):
        return value
    if value is None:
        return ToolResult.ok("")
    if isinstance(value, str):
        return ToolResult.ok(value)
    if isinstance(value, BaseModel):
        return ToolResult.ok(value.model_dump_json(), data=value.model_dump(mode="json"))
    try:
        return ToolResult.ok(json.dumps(value, default=str, ensure_ascii=False), data=value)
    except (TypeError, ValueError):
        return ToolResult.ok(str(value), data=value)


def load_custom_tool(name: str, options: dict[str, Any]) -> CallableTool:
    unknown = set(options) - CUSTOM_OPTIONS
    if unknown:
        raise CustomToolError(f"tool {name!r}: unknown option(s) {', '.join(sorted(unknown))} (a custom tool takes: {', '.join(sorted(CUSTOM_OPTIONS))})")
    side_effects = options.get("side_effects", "none")
    if side_effects not in SIDE_EFFECTS:
        raise CustomToolError(f"tool {name!r}: side_effects must be one of {', '.join(SIDE_EFFECTS)}, not {side_effects!r}")
    timeout = options.get("timeout_s")
    if timeout is not None and not (isinstance(timeout, int | float) and timeout > 0):
        raise CustomToolError(f"tool {name!r}: timeout_s must be a positive number")
    function = import_callable(str(options["path"]))
    return CallableTool(name, function, description=options.get("description"), side_effects=side_effects, timeout_s=timeout)
