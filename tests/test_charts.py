"""The report's SVG charts: valid XML, no external references, escaped text, and sensible handling of missing, degenerate and many-point data."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET

import pytest

from ragbench.reporting.charts import (
    Point,
    frontier,
    heatmap_svg,
    latency_bars_svg,
    nice_ticks,
    scatter_svg,
    stacked_bars_svg,
    whisker_svg,
)

NS = {"s": "http://www.w3.org/2000/svg"}
HOSTILE = '<script>alert("x")</script> & "q"'


def _parse(svg: str) -> ET.Element:
    root = ET.fromstring(svg)  # raises on anything that is not well-formed XML
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    return root


def _no_external_references(svg: str) -> None:
    assert not re.search(r"(href|src|url\()\s*=?\s*[\"']?(https?:|//)", svg)
    assert "http://" not in svg.replace("http://www.w3.org/2000/svg", "") and "https://" not in svg


def _points() -> list[Point]:
    return [
        Point("bm25", 0.0004, 4.0, 5.0, "winner", "bm25 · 4.0"),
        Point("vector", 0.0006, 3.2, 7.0, "other", "vector · 3.2"),
        Point("hybrid_rerank", 0.0030, 4.1, 60.0, "pareto", "hybrid_rerank · 4.1"),
        Point("llm_heavy", 0.0200, 4.2, 900.0, "tie", "llm_heavy · 4.2"),
    ]


def test_nice_ticks_are_round_and_cover_the_range():
    ticks = nice_ticks(0.0, 4.7)
    assert ticks[0] <= 0.0 and ticks[-1] >= 4.0 and all(abs(t * 2 - round(t * 2)) < 1e-9 or abs(t * 4 - round(t * 4)) < 1e-9 for t in ticks)
    assert len(nice_ticks(5.0, 5.0)) >= 2 and nice_ticks(3, 1) == nice_ticks(1, 3)
    assert len(nice_ticks(0.0003, 0.0210)) >= 3


def test_scatter_is_valid_svg_with_one_focusable_mark_per_system_and_a_tooltip_on_each():
    svg = scatter_svg(_points(), x_label="Cost per question", y_label="Answer score", x_fmt=lambda v: f"${v:.4f}", y_fmt=lambda v: f"{v:.1f}")
    root = _parse(svg)
    marks = root.findall(".//s:g[@class]", NS)
    marks = [m for m in marks if "mark" in m.get("class", "").split()]
    assert len(marks) == 4 and all(m.get("tabindex") == "0" and m.get("data-tip") and m.get("aria-label") for m in marks)
    assert {m.get("class").split()[1] for m in marks} == {"mk-winner", "mk-tie", "mk-pareto", "mk-other"}
    assert root.find(".//s:path[@class='frontier']", NS) is not None
    texts = " ".join(t.text or "" for t in root.iter("{http://www.w3.org/2000/svg}text"))
    assert "bm25" in texts and "hybrid_rerank" in texts and "vector" not in texts  # the muted system is not labeled; hover and the table carry it
    _no_external_references(svg)


def test_scatter_uses_a_log_axis_for_costs_that_span_orders_of_magnitude_and_says_so():
    spread = scatter_svg(_points(), x_label="Cost", y_label="Score", x_fmt=str, y_fmt=str)
    assert "log scale" in spread
    flat = scatter_svg(_points(), x_label="Cost", y_label="Score", x_fmt=str, y_fmt=str, log_x=False)
    assert "log scale" not in flat


@pytest.mark.parametrize("count", [1, 2, 18])
def test_scatter_handles_one_two_and_many_systems(count):
    points = [Point(f"system_{n}", 0.001 * (n + 1), 3 + (n % 5) / 4, 10.0 * (n + 1), ["winner", "pareto", "tie", "other"][n % 4], f"tip {n}") for n in range(count)]
    root = _parse(scatter_svg(points, x_label="x", y_label="y", x_fmt=str, y_fmt=str))
    assert len([m for m in root.iter("{http://www.w3.org/2000/svg}g") if "mark" in m.get("class", "")]) == count


def test_scatter_with_identical_or_missing_values_does_not_divide_by_zero():
    same = [Point("a", 0.0, 4.0, None, "winner", "a"), Point("b", 0.0, 4.0, None, "other", "b")]
    _parse(scatter_svg(same, x_label="x", y_label="y", x_fmt=str, y_fmt=str))
    missing = [Point("a", None, 4.0, None, "other", "a"), Point("b", 1.0, None, None, "other", "b")]
    root = _parse(scatter_svg(missing, x_label="x", y_label="y", x_fmt=str, y_fmt=str))
    assert not [m for m in root.iter("{http://www.w3.org/2000/svg}g") if "mark" in m.get("class", "")]
    _parse(scatter_svg([], x_label="x", y_label="y", x_fmt=str, y_fmt=str))


def test_scatter_escapes_system_names():
    svg = scatter_svg([Point(HOSTILE, 1.0, 3.0, 5.0, "winner", HOSTILE)], x_label="x", y_label="y", x_fmt=str, y_fmt=str)
    _parse(svg)
    assert "<script>" not in svg


def test_the_frontier_keeps_only_systems_no_other_system_beats_on_both_axes():
    names = [p.name for p in frontier(_points())]
    assert names == ["bm25", "hybrid_rerank", "llm_heavy"] and "vector" not in names
    assert frontier([Point("a", None, 1.0, None, "other", "")]) == []


def test_heatmap_marks_missing_cells_as_missing_not_zero():
    svg = heatmap_svg(["bm25", "vector"], ["single_hop", "numeric", "multi_hop"], [[4.5, 3.0, None], [2.0, float("nan"), 4.8]], col_notes=["10", "4", "6"], value_name="answer score")
    root = _parse(svg)
    cells = [g for g in root.iter("{http://www.w3.org/2000/svg}g") if "cell" in g.get("class", "").split()]
    assert len(cells) == 6 and sum("na" in g.get("class").split() for g in cells) == 2
    assert all("no data" in g.get("data-tip") for g in cells if "na" in g.get("class").split())
    assert any("c6" in g.get("class").split() for g in cells) and any("c0" in g.get("class").split() for g in cells)  # the scale spans the values shown
    assert "n=10" in svg
    _no_external_references(svg)


def test_heatmap_with_no_data_or_one_value_still_renders():
    _parse(heatmap_svg(["a"], ["x"], [[None]]))
    _parse(heatmap_svg(["a", "b"], ["x"], [[3.0], [3.0]]))
    _parse(heatmap_svg([], [], []))


def test_heatmap_escapes_labels():
    svg = heatmap_svg([HOSTILE], [HOSTILE], [[1.0]])
    _parse(svg)
    assert "<script>" not in svg


def test_stacked_bars_split_each_row_with_gaps_and_write_the_total_at_the_tip():
    rows = [("bm25", {"retrieve": 0.001, "llm": 0.003}), ("vector", {"retrieve": 0.002, "embed": 0.0005, "llm": 0.004}), ("free", {})]
    svg = stacked_bars_svg(rows, ["retrieve", "embed", "llm"], {"retrieve": "s1", "embed": "s5", "llm": "s3"}, fmt=lambda v: f"${v:.4f}", value_name="cost")
    root = _parse(svg)
    segments = [g for g in root.iter("{http://www.w3.org/2000/svg}g") if "seg" in g.get("class", "").split()]
    assert len(segments) == 5  # two + three + none: an all-zero row is an empty track
    assert any("$0.0040" in (t.text or "") for t in root.iter("{http://www.w3.org/2000/svg}text"))
    classes = {p.get("class") for p in root.iter("{http://www.w3.org/2000/svg}path")}
    assert {"s1", "s5", "s3"} <= classes
    assert all(g.get("data-tip") and "%" in g.get("data-tip") for g in segments)
    _no_external_references(svg)


def test_stacked_bars_with_tiny_values_keep_a_visible_hoverable_segment():
    svg = stacked_bars_svg([("a", {"x": 1e-9, "y": 1.0})], ["x", "y"], {"x": "s1", "y": "s2"}, fmt=lambda v: f"{v:g}")
    root = _parse(svg)
    widths = [float(r.get("width")) for r in root.iter("{http://www.w3.org/2000/svg}rect") if r.get("class") == "hit"]
    assert min(widths) >= 2


def test_latency_bars_show_p50_inside_p95_and_tolerate_missing_values():
    svg = latency_bars_svg([("bm25", 5.0, 7.0), ("vector", 6.5, 9.0), ("slow", None, None), ("half", 12.0, None)])
    root = _parse(svg)
    texts = [t.text for t in root.iter("{http://www.w3.org/2000/svg}text")]
    assert "5 / 7 ms" in texts and "no latency recorded" in texts and "12 ms" in texts
    assert len([p for p in root.iter("{http://www.w3.org/2000/svg}path") if p.get("class") == "lat-p50"]) == 3
    _parse(latency_bars_svg([]))


def test_whisker_is_a_small_valid_svg_or_nothing_without_an_interval():
    svg = whisker_svg(4.0, 3.1, 4.7, 0.0, 5.0)
    root = _parse(svg)
    assert root.get("aria-hidden") == "true" and root.find(".//s:circle", NS) is not None
    assert whisker_svg(4.0, None, 4.7, 0, 5) == "" and whisker_svg(float("nan"), 1, 2, 0, 5) == "" and whisker_svg(1, 0, 2, 3, 3) == ""
    far = _parse(whisker_svg(9.0, 8.0, 10.0, 0.0, 5.0))  # values outside the scale are clamped into the frame
    assert all(0 <= float(c.get("cx")) <= 76 for c in far.iter("{http://www.w3.org/2000/svg}circle"))


def test_no_chart_writes_a_color_into_the_markup():
    svgs = [
        scatter_svg(_points(), x_label="x", y_label="y", x_fmt=str, y_fmt=str),
        heatmap_svg(["a"], ["b"], [[1.0]]),
        stacked_bars_svg([("a", {"x": 1.0})], ["x"], {"x": "s1"}, fmt=str),
        latency_bars_svg([("a", 1.0, 2.0)]),
        whisker_svg(1, 0, 2, 0, 3),
    ]
    for svg in svgs:  # colors live in the stylesheet so one chart serves light, dark and print
        assert not re.search(r"#[0-9a-fA-F]{3,8}\b|rgb\(|hsl\(", svg)
        assert not math.isnan(len(svg))
