"""A scripted agent harness for the `cli_agent` tests: replays a recorded event stream (`tests/fixtures/cli_agent/*.jsonl`), with
`{workspace}` replaced by the directory it runs in, and can misbehave on request.

    python tests/fake_cli_agent.py FIXTURE [--exit-code N] [--stderr TEXT] [--sleep S] [--repeat-tool N] [--answer-file PATH --answer TEXT] [--dump FILE]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture")
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--stderr", default="")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--repeat-tool", type=int, default=0, help="before the fixture, print this many Claude Code Grep calls (for the step cap)")
    parser.add_argument("--answer-file", default=None)
    parser.add_argument("--answer", default="")
    parser.add_argument("--dump", default=None, help="write the argv, the cwd, the environment and every workspace file to this JSON file")
    parser.add_argument("rest", nargs="*")
    args = parser.parse_args()
    workspace = Path.cwd()
    if args.dump:
        files = {path.relative_to(workspace).as_posix(): path.read_text(encoding="utf-8") for path in sorted(workspace.rglob("*")) if path.is_file()}
        writable = os.access(workspace / "docs", os.W_OK)
        Path(args.dump).write_text(json.dumps({"argv": sys.argv[1:], "cwd": str(workspace), "env": dict(os.environ), "files": files, "docs_writable": writable}), encoding="utf-8")
    for index in range(args.repeat_tool):
        use = {"type": "tool_use", "id": f"toolu_r{index}", "name": "Grep", "input": {"pattern": f"term{index}", "path": "docs"}}
        print(json.dumps({"type": "assistant", "message": {"id": f"msg_r{index}", "content": [use], "usage": {"input_tokens": 100, "output_tokens": 10}}}), flush=True)
        print(json.dumps({"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": use["id"], "content": "No matches found"}]}}), flush=True)
        time.sleep(0.01)
    if args.sleep:
        time.sleep(args.sleep)
    if args.answer_file:
        (workspace / args.answer_file).write_text(args.answer, encoding="utf-8")
    text = Path(args.fixture).read_text(encoding="utf-8") if args.fixture != "-" else ""
    sys.stdout.write(text.replace("{workspace}", workspace.as_posix()))
    sys.stdout.flush()
    if args.stderr:
        sys.stderr.write(args.stderr)
    return args.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
