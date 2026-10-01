"""SVG charts for the HTML report: pure functions from numbers to an SVG string, no I/O, no external references.

Colors are never written into the SVG. Marks carry classes (`s1`..`s8` categorical slots, `sgray`, `c0`..`c6` sequential steps, `mk-*` roles) and the
report's stylesheet maps them to the light and dark palettes, so one SVG serves both themes. Every interactive mark is a `<g class="mark" tabindex="0"
data-tip="...">`: the report's script shows `data-tip` as a tooltip on hover and keyboard focus (and every value is also in the table view next to the
chart, so a tooltip never gates a number). Text from the data (system and category names) is escaped; nothing is interpolated unescaped.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from xml.sax.saxutils import escape, quoteattr

SVG_NS = "http://www.w3.org/2000/svg"
CHAR_W = 6.2  # average glyph width at the chart font size (11px), for label fitting; labels are placed conservatively
MAX_BAR = 22  # bars are never thicker than this
STEPS = 7  # sequential steps (`c0`..`c6`)
MAX_LABELED = 7  # labels beside bubbles besides the winner's: more than this is clutter, and hover and the table have the rest
Number = float | int


def _clean(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) or math.isinf(number) else number


def _f(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".") if abs(value) < 1e6 else f"{value:.0f}"


def _svg(width: float, height: float, label: str, body: str, css_class: str, min_width: float | None = None) -> str:
    """The chart's root element. On a narrow screen the chart keeps `min_width` pixels (its container scrolls) instead of shrinking its text to nothing."""
    keep = min(width, 700) if min_width is None else min_width
    return (
        f'<svg xmlns="{SVG_NS}" class="chart {css_class}" viewBox="0 0 {_f(width)} {_f(height)}" style="min-width:{_f(keep)}px" role="img" aria-label={quoteattr(label)}>'
        f"<title>{escape(label)}</title>{body}</svg>"
    )


def _text(x: float, y: float, text: str, css: str = "", anchor: str = "start", extra: str = "") -> str:
    return f'<text x="{_f(x)}" y="{_f(y)}" class="{css}" text-anchor="{anchor}"{extra}>{escape(text)}</text>'


def nice_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    """Round tick values covering [lo, hi]; a degenerate range still gets two ticks."""
    if hi < lo:
        lo, hi = hi, lo
    if hi == lo:
        pad = abs(hi) * 0.1 or 1.0
        lo, hi = lo - pad, hi + pad
    span = (hi - lo) / max(1, count)
    magnitude = 10 ** math.floor(math.log10(span))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= span)
    first = math.floor(lo / step) * step
    ticks: list[float] = []
    value = first
    while value <= hi + step * 0.5 and len(ticks) < 40:
        if value >= lo - step * 1e-9:
            ticks.append(round(value, 12))
        value += step
    return ticks or [lo, hi]


# -- scatter / bubble ------------------------------------------------------------------------------


@dataclass
class Point:
    name: str
    x: float | None
    y: float | None
    size: float | None  # bubble area encodes this (latency); None draws the smallest bubble
    kind: str  # "winner" | "tie" | "pareto" | "other": how the mark looks (emphasis, not one hue per system)
    tip: str


def frontier(points: Sequence[Point]) -> list[Point]:
    """The cost-quality frontier: points no other point beats on both (x lower is better, y higher is better), ordered by x."""
    usable = sorted((p for p in points if p.x is not None and p.y is not None), key=lambda p: (p.x, -p.y))  # type: ignore[operator]
    best, kept = -math.inf, []
    for point in usable:
        if point.y is not None and point.y > best:
            kept.append(point)
            best = point.y
    return kept


def scatter_svg(
    points: Sequence[Point],
    *,
    x_label: str,
    y_label: str,
    x_fmt: Callable[[float], str],
    y_fmt: Callable[[float], str],
    log_x: bool | None = None,
    width: int = 640,
    height: int = 400,
) -> str:
    """Quality (y) against cost or latency (x), bubble area = latency. Winner, ties and Pareto-optimal systems are drawn and labeled; the rest are muted.

    A staircase through the cost-quality frontier shows what an extra dollar buys. `log_x=None` picks a log axis when costs span more than 30x.
    """
    left, right, top, bottom = 66, 28, 18, 56
    plot_w, plot_h = width - left - right, height - top - bottom
    drawn = [p for p in points if p.x is not None and p.y is not None]
    xs = [p.x for p in drawn if p.x is not None]
    ys = [p.y for p in drawn if p.y is not None]
    log = bool(xs) and min(xs) > 0 and max(xs) / min(xs) >= 30 if log_x is None else log_x
    if xs:
        lo_x, hi_x = (min(xs), max(xs)) if log else (0.0, max(xs))
    else:
        lo_x, hi_x = 0.0, 1.0
    if hi_x <= lo_x:
        lo_x, hi_x = (lo_x / 2, hi_x * 2) if log and lo_x > 0 else (0.0, 1.0)
    pad = (hi_x - lo_x) * 0.06 if not log else 0
    x_lo, x_hi = (lo_x / 1.4, hi_x * 1.4) if log else (lo_x, hi_x + pad)
    lo_y, hi_y = (min(ys), max(ys)) if ys else (0.0, 1.0)
    margin = (hi_y - lo_y) * 0.12 or 0.5
    y_lo, y_hi = lo_y - margin, hi_y + margin

    def sx(value: float) -> float:
        if log:
            return left + plot_w * (math.log(value) - math.log(x_lo)) / (math.log(x_hi) - math.log(x_lo))
        return left + plot_w * (value - x_lo) / (x_hi - x_lo)

    def sy(value: float) -> float:
        return top + plot_h * (1 - (value - y_lo) / (y_hi - y_lo))

    parts: list[str] = []
    for tick in nice_ticks(y_lo, y_hi, 5):
        if y_lo <= tick <= y_hi:
            parts.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{_f(sy(tick))}" y2="{_f(sy(tick))}"/>')
            parts.append(_text(left - 8, sy(tick) + 3.5, y_fmt(tick), "tick", "end"))
    if log:
        decade = 10 ** math.floor(math.log10(x_lo))
        x_ticks: list[float] = []
        while decade <= x_hi:
            x_ticks.extend(m * decade for m in (1, 3) if x_lo <= m * decade <= x_hi)
            decade *= 10
    else:
        x_ticks = [t for t in nice_ticks(x_lo, x_hi, 5) if x_lo <= t <= x_hi]
    for tick in x_ticks:
        parts.append(f'<line class="grid" x1="{_f(sx(tick))}" x2="{_f(sx(tick))}" y1="{top}" y2="{top + plot_h}"/>')
        parts.append(_text(sx(tick), top + plot_h + 18, x_fmt(tick), "tick", "middle"))
    parts.append(f'<line class="axis" x1="{left}" x2="{width - right}" y1="{top + plot_h}" y2="{top + plot_h}"/>')
    parts.append(_text(left + plot_w / 2, height - 8, x_label + (" (log scale)" if log else ""), "axis-title", "middle"))
    parts.append(f'<text class="axis-title" text-anchor="middle" transform="translate(14 {_f(top + plot_h / 2)}) rotate(-90)">{escape(y_label)}</text>')

    front = frontier(drawn)
    if len(front) >= 2:
        path = f"M{_f(sx(front[0].x))},{_f(sy(front[0].y))}"  # type: ignore[arg-type]
        for point in front[1:]:
            path += f" H{_f(sx(point.x))} V{_f(sy(point.y))}"  # type: ignore[arg-type]
        parts.append(f'<path class="frontier" d="{path}" fill="none"/>')

    sizes = [p.size for p in drawn if p.size is not None]
    s_lo, s_hi = (min(sizes), max(sizes)) if sizes else (0.0, 0.0)

    def radius(size: float | None) -> float:
        if size is None or s_hi <= s_lo:
            return 6.0
        return 4.5 + 8.5 * math.sqrt((size - s_lo) / (s_hi - s_lo))

    order = {"other": 0, "pareto": 1, "tie": 2, "winner": 3}
    marks: list[str] = []
    labels: list[str] = []
    taken: list[tuple[float, float, float, float]] = []
    for point in sorted(drawn, key=lambda p: order.get(p.kind, 0)):
        cx, cy, r = sx(point.x), sy(point.y), radius(point.size)  # type: ignore[arg-type]
        marks.append(
            f'<g class="mark mk-{point.kind}" tabindex="0" data-tip={quoteattr(point.tip)} aria-label={quoteattr(point.tip)}>'
            f'<circle class="hit" cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(max(r + 4, 14))}"/><circle class="dot" cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(r)}"/></g>'
        )
    dots = [(sx(p.x), sy(p.y), radius(p.size)) for p in drawn]  # type: ignore[arg-type]
    labeled_others = 0
    for point in sorted(drawn, key=lambda p: -order.get(p.kind, 0)):
        if point.kind == "other" or (point.kind != "winner" and labeled_others >= MAX_LABELED):
            continue
        cx, cy, r = sx(point.x), sy(point.y), radius(point.size)  # type: ignore[arg-type]
        w = len(point.name) * CHAR_W + 4
        for anchor, lx, ly in (("start", cx + r + 6, cy + 4), ("end", cx - r - 6, cy + 4), ("middle", cx, cy - r - 7), ("middle", cx, cy + r + 15)):
            x0 = lx if anchor == "start" else lx - w if anchor == "end" else lx - w / 2
            box = (x0, ly - 11, x0 + w, ly + 3)
            if box[0] < 2 or box[2] > width - 2 or box[1] < 2 or box[3] > top + plot_h + 2:
                continue
            if any(not (box[2] < t[0] or box[0] > t[2] or box[3] < t[1] or box[1] > t[3]) for t in taken):
                continue
            if any(not (box[2] < dx - dr - 2 or box[0] > dx + dr + 2 or box[3] < dy - dr - 2 or box[1] > dy + dr + 2) for dx, dy, dr in dots):
                continue  # a label never sits on another system's bubble
            taken.append(box)
            labels.append(_text(lx, ly, point.name, "label strong" if point.kind == "winner" else "label", anchor))
            labeled_others += point.kind != "winner"
            break
    return _svg(width, height, f"{y_label} against {x_label}, one mark per system", "".join(parts + marks + labels), "scatter")


# -- heatmap ---------------------------------------------------------------------------------------


def _bucket(value: float, lo: float, hi: float) -> int:
    if hi <= lo:
        return STEPS // 2
    return min(STEPS - 1, max(0, int((value - lo) / (hi - lo) * STEPS)))


def heatmap_svg(
    rows: Sequence[str],
    cols: Sequence[str],
    values: Sequence[Sequence[object]],
    *,
    col_notes: Sequence[str] | None = None,
    fmt: str = "{:.2f}",
    value_name: str = "value",
    vmin: float | None = None,
    vmax: float | None = None,
) -> str:
    """Rows (systems) by columns (categories), shaded by `value` on one sequential ramp (darker = higher; the scale spans the values shown).

    A missing value (None or NaN) is an empty hatched cell with a dash, never a 0. Cells with room carry their value; every cell has a tooltip.
    """
    cleaned = [[_clean(v) for v in row] for row in values]
    flat = [v for row in cleaned for v in row if v is not None]
    lo = vmin if vmin is not None else (min(flat) if flat else 0.0)
    hi = vmax if vmax is not None else (max(flat) if flat else 1.0)
    label_w = min(220, max(110, max((len(r) for r in rows), default=8) * CHAR_W + 14))
    cell_w, cell_h, header_h = 62, 26, 92
    width = label_w + cell_w * max(1, len(cols)) + 96  # room for the last rotated column label
    height = header_h + cell_h * len(rows) + 8
    parts = []
    for j, name in enumerate(cols):
        x = label_w + j * cell_w + cell_w / 2
        note = f" · n={col_notes[j]}" if col_notes else ""
        parts.append(
            f'<text class="col-label" transform="translate({_f(x)} {header_h - 8}) rotate(-38)" text-anchor="start">{escape(name)}'
            f'<tspan class="muted">{escape(note)}</tspan></text>'
        )
    for i, row_name in enumerate(rows):
        y = header_h + i * cell_h
        parts.append(_text(label_w - 8, y + cell_h / 2 + 4, row_name, "row-label", "end"))
        for j, col in enumerate(cols):
            value = cleaned[i][j] if j < len(cleaned[i]) else None
            x = label_w + j * cell_w
            tip = f"{row_name} · {col}: " + (fmt.format(value) if value is not None else "no data")
            if value is None:
                parts.append(
                    f'<g class="mark cell na" tabindex="0" data-tip={quoteattr(tip)} aria-label={quoteattr(tip)}>'
                    f'<rect x="{x + 1}" y="{y + 1}" width="{cell_w - 2}" height="{cell_h - 2}" rx="3"/>{_text(x + cell_w / 2, y + cell_h / 2 + 4, "—", "cell-text na", "middle")}</g>'
                )
                continue
            bucket = _bucket(value, lo, hi)
            parts.append(
                f'<g class="mark cell c{bucket}" tabindex="0" data-tip={quoteattr(tip)} aria-label={quoteattr(tip)}>'
                f'<rect x="{x + 1}" y="{y + 1}" width="{cell_w - 2}" height="{cell_h - 2}" rx="3"/>'
                f'{_text(x + cell_w / 2, y + cell_h / 2 + 4, fmt.format(value), "cell-text " + ("lo" if bucket < 4 else "hi"), "middle")}</g>'
            )
    return _svg(width, height, f"{value_name} by system and category", "".join(parts), "heatmap", min_width=width)


# -- horizontal stacked bars -----------------------------------------------------------------------


def _round_end(x: float, y: float, w: float, h: float, r: float = 4) -> str:
    """A bar segment with a rounded right end and a square left end."""
    r = min(r, w, h / 2)
    return f"M{_f(x)},{_f(y)} H{_f(x + w - r)} Q{_f(x + w)},{_f(y)} {_f(x + w)},{_f(y + r)} V{_f(y + h - r)} Q{_f(x + w)},{_f(y + h)} {_f(x + w - r)},{_f(y + h)} H{_f(x)} Z"


def stacked_bars_svg(
    rows: Sequence[tuple[str, dict[str, float | None]]],
    keys: Sequence[str],
    slots: dict[str, str],
    *,
    fmt: Callable[[float], str],
    key_names: dict[str, str] | None = None,
    value_name: str = "total",
    width: int = 640,
) -> str:
    """One horizontal bar per row, split into `keys` (in that order) with a 2px surface gap; the bar total is written at its tip.

    `slots` maps each key to a color class (`s1`..`s8`, `sgray`). Segments too narrow to see are still hoverable. All-zero rows draw an empty track.
    """
    names = key_names or {}
    label_w = min(200, max(100, max((len(r) for r, _ in rows), default=8) * CHAR_W + 14))
    value_w = 74
    plot_w = width - label_w - value_w
    row_h = 32
    totals = [sum(v for v in data.values() if v) for _, data in rows]
    peak = max(totals, default=0.0) or 1.0
    parts = []
    for i, (label, data) in enumerate(rows):
        y = i * row_h + 5
        parts.append(_text(label_w - 8, y + MAX_BAR / 2 + 4, label, "row-label", "end"))
        parts.append(f'<rect class="track" x="{label_w}" y="{y}" width="{plot_w}" height="{MAX_BAR}" rx="4"/>')
        x = float(label_w)
        present = [(key, data.get(key) or 0.0) for key in keys if (data.get(key) or 0.0) > 0]
        for n, (key, value) in enumerate(present):
            w = max(2.0, plot_w * value / peak)
            tip = f"{label} · {names.get(key, key)}: {fmt(value)} ({value / totals[i]:.0%} of {fmt(totals[i])})"
            gap = 2 if n < len(present) - 1 else 0
            shape = (
                _round_end(x, y, w - gap, MAX_BAR)
                if n == len(present) - 1
                else f"M{_f(x)},{_f(y)} h{_f(max(1.0, w - gap))} v{MAX_BAR} h{_f(-max(1.0, w - gap))} Z"
            )
            parts.append(
                f'<g class="mark seg" tabindex="0" data-tip={quoteattr(tip)} aria-label={quoteattr(tip)}>'
                f'<path class="{slots.get(key, "sgray")}" d="{shape}"/><path class="tex t{n % 2}" d="{shape}"/>'
                f'<rect class="hit" x="{_f(x)}" y="{y - 3}" width="{_f(w)}" height="{MAX_BAR + 6}"/></g>'
            )
            x += w
        parts.append(_text(x + 8, y + MAX_BAR / 2 + 4, fmt(totals[i]), "value", "start"))
    height = max(1, len(rows)) * row_h + 10
    defs = (
        '<defs><pattern id="hatch-a" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" class="hatch-line"/></pattern>'
        '<pattern id="hatch-b" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(135)"><line x1="0" y1="0" x2="0" y2="6" class="hatch-line"/></pattern></defs>'
    )
    return _svg(width, height, f"{value_name} by system, split into parts", defs + "".join(parts), "stacked")


# -- latency bars ----------------------------------------------------------------------------------


def latency_bars_svg(rows: Sequence[tuple[str, object, object]], *, width: int = 640) -> str:
    """Per system a long light bar to p95 with a short dark bar to p50 on top (one hue, two shades); 'p50 / p95 ms' is written once at the tip."""
    label_w = min(200, max(100, max((len(r) for r, _, _ in rows), default=8) * CHAR_W + 14))
    value_w = 104
    plot_w = width - label_w - value_w
    row_h = 32
    cleaned = [(name, _clean(p50), _clean(p95)) for name, p50, p95 in rows]
    peak = max((v for _, a, b in cleaned for v in (a, b) if v is not None), default=0.0) or 1.0
    parts = []
    for i, (name, p50, p95) in enumerate(cleaned):
        y = i * row_h + 5
        parts.append(_text(label_w - 8, y + MAX_BAR / 2 + 4, name, "row-label", "end"))
        parts.append(f'<rect class="track" x="{label_w}" y="{y}" width="{plot_w}" height="{MAX_BAR}" rx="4"/>')
        if p95 is None and p50 is None:
            parts.append(_text(label_w + 8, y + MAX_BAR / 2 + 4, "no latency recorded", "value"))
            continue
        long_end = max(p95 if p95 is not None else p50 or 0.0, p50 or 0.0)
        tip = f"{name} · p50 {p50:.0f} ms · p95 {p95:.0f} ms" if p50 is not None and p95 is not None else f"{name} · {long_end:.0f} ms"
        w95, w50 = max(2.0, plot_w * long_end / peak), max(2.0, plot_w * (p50 or 0.0) / peak)
        parts.append(
            f'<g class="mark seg" tabindex="0" data-tip={quoteattr(tip)} aria-label={quoteattr(tip)}>'
            f'<path class="lat-p95" d="{_round_end(label_w, y, w95, MAX_BAR)}"/><path class="lat-p50" d="{_round_end(label_w, y + 6, w50, MAX_BAR - 12, 3)}"/>'
            f'<rect class="hit" x="{label_w}" y="{y - 3}" width="{_f(w95)}" height="{MAX_BAR + 6}"/></g>'
        )
        text = f"{p50:.0f} / {p95:.0f} ms" if p50 is not None and p95 is not None else f"{long_end:.0f} ms"
        parts.append(_text(label_w + w95 + 8, y + MAX_BAR / 2 + 4, text, "value"))
    return _svg(width, max(1, len(rows)) * row_h + 10, "Latency per system: p50 and p95", "".join(parts), "latency")


# -- confidence whisker (inline in table cells) ----------------------------------------------------


def whisker_svg(value: object, lo: object, hi: object, scale_lo: float, scale_hi: float, *, width: int = 76, height: int = 10) -> str:
    """A 95% interval as a hairline with a dot at the mean, on the column's own scale. Empty string when there is no interval."""
    v, a, b = _clean(value), _clean(lo), _clean(hi)
    if v is None or a is None or b is None or scale_hi <= scale_lo:
        return ""

    def sx(number: float) -> float:
        return 4 + (width - 8) * (min(max(number, scale_lo), scale_hi) - scale_lo) / (scale_hi - scale_lo)

    return (
        f'<svg xmlns="{SVG_NS}" class="whisker" viewBox="0 0 {width} {height}" width="{width}" height="{height}" aria-hidden="true">'
        f'<line class="whisker-line" x1="{_f(sx(a))}" x2="{_f(sx(b))}" y1="{height / 2}" y2="{height / 2}"/>'
        f'<line class="whisker-cap" x1="{_f(sx(a))}" x2="{_f(sx(a))}" y1="2" y2="{height - 2}"/><line class="whisker-cap" x1="{_f(sx(b))}" x2="{_f(sx(b))}" y1="2" y2="{height - 2}"/>'
        f'<circle class="whisker-dot" cx="{_f(sx(v))}" cy="{height / 2}" r="3"/></svg>'
    )
