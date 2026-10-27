"""Render the README's 20-second GIF: `ragbench auto` on the bundled demo documents, then the report it wrote.

    python scripts/make_demo_gif.py            # writes docs/assets/auto-to-report.gif (needs Pillow: pip install pillow)

It runs `ragbench auto --mock` for real in a temporary folder (no key, no network, about five seconds), keeps the lines that tell the story
(documents profiled, questions written, each system finishing, the recommendation) and types them out like a terminal; the last scene is
the top of the report of the published MultiHop-RAG run (`docs/benchmarks/multihop-rag-run/report-desktop.png`), which is what a real run's report
looks like. Nothing in the GIF is typed by hand except the command line.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "auto-to-report.gif"
REPORT = ROOT / "docs" / "benchmarks" / "multihop-rag-run" / "report-desktop.png"
COMMAND = "ragbench auto --docs ./my_docs --mock"
COLUMNS = 100
FONT = "/System/Library/Fonts/Menlo.ttc"
SIZE = 15
BG, FG, DIM, ACCENT, GREEN = (22, 24, 29), (220, 223, 228), (128, 134, 145), (125, 176, 255), (130, 214, 150)
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def capture() -> list[str]:
    """Run `ragbench auto --mock` on the demo documents and return its output, ANSI removed."""
    work = Path(tempfile.mkdtemp(prefix="ragbench-gif-"))
    try:
        env = {**os.environ, "COLUMNS": str(COLUMNS), "NO_COLOR": "1"}
        run = subprocess.run(
            [str(Path(sys.executable).with_name("ragbench")), "auto", "--docs", str(ROOT / "data" / "demo" / "docs"), "--mock", "--output-dir", str(work / "results")],
            cwd=work, env=env, capture_output=True, text=True, check=True,
        )
        return [ANSI.sub("", line).rstrip() for line in run.stdout.splitlines()]
    finally:
        shutil.rmtree(work, ignore_errors=True)


def story(lines: list[str]) -> list[tuple[str, tuple[int, int, int]]]:
    """The lines worth showing, coloured: profile, question count, one line per finished system, the recommendation box, where the files are."""
    shown: list[tuple[str, tuple[int, int, int]]] = []
    in_box = False
    for line in lines:
        if re.match(r"^\d+ documents, ", line) or re.match(r"^\d+ questions \(", line):
            shown.append((line, FG))
        elif " · done in " in line:
            shown.append((line.split("━")[0].rstrip() + "  ✓", GREEN))
        elif line.startswith("╭─ Recommendation"):
            in_box = True
            shown.append(("", FG))
        if in_box:
            shown.append((line, ACCENT if line.startswith(("╭", "╰")) else FG))
            in_box = not line.startswith("╰")
        elif line.startswith("Deploy "):
            shown.append(("", FG))
            shown.append((line, GREEN))
        elif re.match(r"^\s+(Runnable config|Report|Reasons): ", line):
            name, path = line.strip().split(": ", 1)
            shown.append((f"  {name}: results/…/{Path(path).name}", DIM))
    return shown


def terminal(font: ImageFont.FreeTypeFont, rows: list[tuple[str, tuple[int, int, int]]], typed: int, height: int) -> Image.Image:
    """One frame: the prompt with the first `typed` characters of the command, then `rows`."""
    cell = int(font.getlength("M"))
    image = Image.new("RGB", (cell * COLUMNS + 40, height), BG)
    draw = ImageDraw.Draw(image)
    step = SIZE + 6
    draw.text((20, 16), "$ ", font=font, fill=DIM)
    draw.text((20 + 2 * cell, 16), COMMAND[:typed], font=font, fill=FG)
    for number, (text, colour) in enumerate(rows):
        draw.text((20, 16 + step * (number + 2)), text, font=font, fill=colour)
    return image


def build() -> None:
    font = ImageFont.truetype(FONT, SIZE)
    rows = story(capture())
    if not any("Recommendation" in text for text, _ in rows):
        raise SystemExit("the run printed no recommendation; nothing to show")
    height = 16 + (SIZE + 6) * (len(rows) + 3)
    frames: list[Image.Image] = []
    delays: list[int] = []
    for typed in range(0, len(COMMAND) + 1, 2):
        frames.append(terminal(font, [], typed, height))
        delays.append(60)
    delays[-1] = 500
    for count in range(1, len(rows) + 1):
        frames.append(terminal(font, rows[:count], len(COMMAND), height))
        delays.append(700 if rows[count - 1][0].startswith("╭") else 260)
    delays[-1] = 4500
    report = Image.open(REPORT).convert("RGB")
    width = frames[0].width
    shot = report.resize((width, int(report.height * width / report.width))).crop((0, 0, width, height))
    frames.append(shot)
    delays.append(6000)
    palette = [frame.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for frame in frames]
    palette[0].save(OUT, save_all=True, append_images=palette[1:], duration=delays, loop=0, optimize=True, disposal=1)
    print(f"{OUT.relative_to(ROOT)}: {len(frames)} frames, {sum(delays) / 1000:.1f} s, {OUT.stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    build()
