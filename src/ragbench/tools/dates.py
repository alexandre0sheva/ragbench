"""`date_calc`: date differences and offsets. "Today" is the run's frozen `evaluation.tools_now`, so answers never depend on the clock."""

from __future__ import annotations

import calendar
import re
from datetime import date, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, Field

from ragbench.registry import TOOLS
from ragbench.tools.base import BaseTool, ToolContext, ToolResult, make_spec

DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y%m%d",
    "%B %d, %Y",
    "%b %d, %Y",
    "%B %d %Y",
    "%b %d %Y",
    "%d %B %Y",
    "%d %b %Y",
    "%d %B, %Y",
    "%m/%d/%Y",  # US order; a slash date is never read day-first
)
_ORDINAL = re.compile(r"(?<=\d)(st|nd|rd|th)\b", re.I)
MAX_OFFSET = 1_000_000  # days / weeks / months / years: keeps results inside the supported year range


class DateError(ValueError):
    pass


def parse_date(text: str, today: date) -> date:
    """`today` / `now`, an ISO date or datetime, or a common written form (`March 5, 2025`, `5 Mar 2025`, `03/05/2025`)."""
    cleaned = _ORDINAL.sub("", " ".join(text.strip().split())).replace("Sept ", "Sep ")
    if cleaned.lower() in {"today", "now"}:
        return today
    try:
        return datetime.fromisoformat(cleaned.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    raise DateError(f"could not read the date {text!r}; use YYYY-MM-DD")


def add_months(day: date, months: int) -> date:
    """Move by whole calendar months; a day past the end of the target month lands on its last day (Jan 31 + 1 month = Feb 28/29)."""
    index = day.year * 12 + day.month - 1 + months
    year, month = divmod(index, 12)
    if not 1 <= year + 0 <= 9999:
        raise DateError("the result is outside the supported years")
    return date(year, month + 1, min(day.day, calendar.monthrange(year, month + 1)[1]))


def calendar_difference(start: date, end: date) -> tuple[int, int, int]:
    """Whole years, months and remaining days from `start` to `end` (`start <= end`)."""
    months = (end.year - start.year) * 12 + end.month - start.month
    if end.day < start.day:
        months -= 1
    anchor = add_months(start, months)
    return months // 12, months % 12, (end - anchor).days


def _plural(count: int, unit: str) -> str:
    return f"{count} {unit}{'' if count == 1 else 's'}"


class DateArgs(BaseModel):
    operation: Literal["diff", "add", "weekday", "today"] = Field(
        description="`diff`: days from `date` to `other_date`. `add`: `date` plus the given offsets. `weekday`: the weekday of `date`. `today`: the current date."
    )
    date: str | None = Field(default=None, description="A date: YYYY-MM-DD, `March 5, 2025`, or `today`. The start for `diff`, the base for `add` (default today).")
    other_date: str | None = Field(default=None, description="`diff` only: the end date.")
    days: int = Field(default=0, description="`add` only: days to add (negative to subtract).")
    weeks: int = Field(default=0, description="`add` only: weeks to add.")
    months: int = Field(default=0, description="`add` only: calendar months to add.")
    years: int = Field(default=0, description="`add` only: years to add.")


@TOOLS.register("date_calc")
class DateCalcTool(BaseTool):
    Args = DateArgs
    spec = make_spec(
        "date_calc",
        "Date arithmetic: the difference between two dates, a date plus or minus an offset, or the weekday of a date. `today` is a fixed date for the whole run.",
        DateArgs,
    )

    def _run(self, args: DateArgs, ctx: ToolContext) -> ToolResult:
        today = ctx.today
        try:
            if args.operation == "today":
                return ToolResult.ok(f"Today is {today.isoformat()} ({today:%A}).", data=today.isoformat())
            if args.operation == "weekday":
                day = self._need(args.date, "date", today)
                return ToolResult.ok(f"{day.isoformat()} is a {day:%A}.", data=f"{day:%A}")
            if args.operation == "diff":
                start, end = self._need(args.date, "date", today), self._need(args.other_date, "other_date", today)
                return self._diff(start, end)
            base = parse_date(args.date, today) if args.date else today
            return self._add(base, args)
        except DateError as exc:
            return ToolResult.fail(str(exc))
        except (OverflowError, ValueError):
            return ToolResult.fail("the result is outside the supported dates")

    @staticmethod
    def _need(value: str | None, name: str, today: date) -> date:
        if not value:
            raise DateError(f"`{name}` is required for this operation")
        return parse_date(value, today)

    @staticmethod
    def _diff(start: date, end: date) -> ToolResult:
        days = (end - start).days
        first, last = (start, end) if days >= 0 else (end, start)
        years, months, rest = calendar_difference(first, last)
        weeks, spare = divmod(abs(days), 7)
        direction = "" if days >= 0 else " (the end date is earlier)"
        text = (
            f"From {start.isoformat()} to {end.isoformat()}: {days} days{direction} = {_plural(weeks, 'week')} {_plural(spare, 'day')}; "
            f"in calendar terms {_plural(years, 'year')}, {_plural(months, 'month')}, {_plural(rest, 'day')}."
        )
        return ToolResult.ok(text, data={"days": days, "weeks": weeks, "years": years, "months": months, "remaining_days": rest})

    @staticmethod
    def _add(base: date, args: DateArgs) -> ToolResult:
        if any(abs(value) > MAX_OFFSET for value in (args.days, args.weeks, args.months, args.years)):
            raise DateError("the offset is too large")
        moved = add_months(base, args.months + args.years * 12) + timedelta(days=args.days + args.weeks * 7)
        return ToolResult.ok(f"{base.isoformat()} plus the offset is {moved.isoformat()} ({moved:%A}).", data=moved.isoformat())
