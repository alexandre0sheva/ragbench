"""Take the report screenshots used in the docs from a run's `report.html`, at true viewport widths and cut where a section ends.

    python scripts/capture_report.py docs/benchmarks/multihop-rag-run --out docs/assets --desktop light dark --mobile
    python scripts/capture_report.py docs/benchmarks/multihop-rag-run --out docs/benchmarks/multihop-rag-run --desktop auto --mobile

Needs Google Chrome (set `CHROME` to another binary). Headless Chrome will not make a window narrower than about 500 px, so a plain
`--window-size=390,...` lays the page out at the minimum width and crops its right edge. Each shot is therefore the report in an `<iframe>` of
exactly the width wanted, which gets the real media queries, and the image is cropped to that iframe. `HEIGHTS` are where the sections of the
current report layout end (the recommendation card, on a desktop and on a phone; the chart after it scrolls sideways on a phone by design, so a still image would show it cut), so no image stops mid-card; check
the images when the report layout changes.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
WIDTHS = {"desktop": 1280, "mobile": 390}
HEIGHTS = {"desktop": 1545, "mobile": 2040}
THEME_LINE = 'var saved = store("ragbench-report-theme");'


def shoot(page: Path, width: int, height: int, out: Path) -> None:
    """Render `page` in an iframe `width` x `height` and save exactly that area to `out`."""
    with tempfile.TemporaryDirectory() as folder:
        wrapper = Path(folder) / "wrapper.html"
        wrapper.write_text(
            f'<!doctype html><body style="margin:0"><iframe src="{page.as_uri()}" width="{width}" height="{height}" style="border:0;display:block"></iframe>',
            encoding="utf-8",
        )
        raw = Path(folder) / "raw.png"
        subprocess.run(
            [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={max(width, 800)},{height + 120}",
             "--force-device-scale-factor=1", "--virtual-time-budget=6000", f"--screenshot={raw}", wrapper.as_uri()],
            check=True, capture_output=True,
        )
        Image.open(raw).crop((0, 0, width, height)).save(out, optimize=True)


def capture(run: Path, out: Path, desktop: list[str], mobile: bool) -> list[Path]:
    report = (run / "report.html").resolve()
    text = report.read_text(encoding="utf-8")
    if THEME_LINE not in text:
        raise SystemExit("report.html no longer has the theme line this script patches to force the dark theme")
    written: list[Path] = []
    with tempfile.TemporaryDirectory() as folder:
        for theme in desktop:
            page = report
            if theme == "dark":  # the report keeps the theme in localStorage, which a file page cannot preset: patch a copy
                page = Path(folder) / "dark.html"
                page.write_text(text.replace(THEME_LINE, 'var saved = "dark";'), encoding="utf-8")
            target = out / ("report-desktop.png" if theme == "auto" else f"report-desktop-{theme}.png")
            shoot(page, WIDTHS["desktop"], HEIGHTS["desktop"], target)
            written.append(target)
        if mobile:
            target = out / "report-mobile.png"
            shoot(report, WIDTHS["mobile"], HEIGHTS["mobile"], target)
            written.append(target)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path, help="a run folder with report.html")
    parser.add_argument("--out", type=Path, required=True, help="where the PNGs go")
    parser.add_argument("--desktop", nargs="*", default=[], choices=["light", "dark", "auto"], help="desktop shots: light and dark, or auto (the OS theme)")
    parser.add_argument("--mobile", action="store_true", help="also a 390 px wide shot")
    args = parser.parse_args()
    for path in capture(args.run, args.out, args.desktop, args.mobile):
        print(path, Image.open(path).size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
