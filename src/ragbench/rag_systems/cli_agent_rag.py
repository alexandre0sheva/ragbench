"""`cli_agent`: an agent harness (Claude Code, Codex, OpenCode, or any command) answering each question in a fresh workspace.

Per question the harness gets a temporary directory holding a read-only copy of the corpus under `docs/`, the task in `TASK.md` and the
harness files the system declares, and nothing else (no reference answers, no labels). Its event stream is parsed into steps: every model
turn, every tool call (with the documents it searched or read), and the answer. The documents it read are its passages, which retrieval is
scored on. Under `--mock` the real harness is never started: a bundled offline harness prints the same event format instead.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import weakref
from contextlib import nullcontext
from pathlib import Path, PurePosixPath
from typing import IO, Any

from ragbench.documents.schema import Document
from ragbench.models import cost as pricing
from ragbench.models.cost import CostBreakdown
from ragbench.models.refs import parse_model_ref
from ragbench.models.spend import note_spend
from ragbench.rag_systems.cli_trajectory import PARSERS, ToolCall, Trajectory, Turn, action_kind, evidence_paths, input_text
from ragbench.rag_systems.external_http_rag import fill_env
from ragbench.rag_systems.external_rag import ExternalRAG, ExternalReply
from ragbench.rag_systems.options import CliAgentOptions
from ragbench.rag_systems.spec import SystemSpec
from ragbench.rag_systems.trace import PREVIEW_CHARS, Step, sum_costs
from ragbench.registry import SYSTEMS
from ragbench.runtime.limiter import RateLimiter
from ragbench.utils.text import truncate

logger = logging.getLogger(__name__)

TASK_FILE = "TASK.md"
DOCS_DIR = "docs"
INSTRUCTIONS_FILE = "AGENTS.md"
MAX_CONTEXT_CHARS = 6000  # of one passage: what the agent read of a document, as the judge will see it
STDERR_TAIL = 600
PATH_KEYS = ("file_path", "filePath", "path", "notebook_path", "dir", "directory")  # where file tools take their path
TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".rst", ".html", ".htm", ".json", ".jsonl", ".csv", ".tsv", ".xml", ".yaml", ".yml"}
DEFAULT_FORMATS = {"claude_code": "claude_code", "codex": "codex", "opencode": "opencode", "custom": "text"}
TASK = """Answer the question below using only the documents in the `docs/` directory of the current directory. Search and read them as you
need; do not use the web or anything outside this directory. Answer concisely and name the documents you used. If the documents do not
contain the answer, say that you cannot find it.

{deliver}

Question: {question}
"""


@SYSTEMS.register("cli_agent")
class CliAgentRAG(ExternalRAG):
    spec = SystemSpec(
        type="cli_agent",
        title="CLI agent harness",
        summary="An agent harness (Claude Code, Codex, OpenCode or any command) answers each question by searching a read-only copy of the corpus in a fresh workspace",
        best_for="Measuring a coding-agent harness as a RAG system: how well, how fast and at what cost it finds the answer in your documents",
        cost_profile="high",
        latency_profile="slow",
        requires_llm=False,
        agentic=True,
        options=CliAgentOptions,
        chunker=None,
        external=True,
    )
    options: CliAgentOptions

    def __init__(self, config, force_mock: bool = False):
        super().__init__(config, force_mock=force_mock)
        opts = self.options
        self.output_format = opts.output_format or DEFAULT_FORMATS[opts.preset]
        self.harness_model = opts.harness_model or _harness_model(opts.preset, opts.model)
        self._instructions = Path(opts.instructions).read_text(encoding="utf-8") if opts.instructions else None
        self._files = {target: Path(source) for target, source in opts.files.items()}
        for target, source in self._files.items():
            if not source.exists():
                raise ValueError(f"{self.name}: files[{target!r}] = {str(source)!r} does not exist")
        limits = opts.limits
        self._limiter = RateLimiter(max_concurrent=limits.max_concurrent_requests, requests_per_minute=limits.requests_per_minute) if limits.is_set else None
        self._corpus: Path | None = None
        self._paths: dict[str, str] = {}  # workspace path under docs/ -> document id
        if opts.preset == "custom" and opts.network == "off" and not force_mock:
            logger.warning("%s: RAGBench cannot switch a custom harness's network off; make sure it cannot reach the web (or set network: on)", self.name)
        for path in [] if force_mock else global_instructions(opts.preset):
            logger.warning("%s: %s is not empty, and the harness may add it to every prompt; empty or move it for a clean benchmark", self.name, path)

    @property
    def network(self) -> str:
        return "unenforced" if self.options.preset == "custom" and self.options.network == "off" else self.options.network

    def _ingest_external(self, documents: list[Document]) -> None:
        """Write the corpus once; each question gets its own copy of it."""
        corpus = Path(tempfile.mkdtemp(prefix="ragbench-corpus-"))
        weakref.finalize(self, _remove, corpus)
        self._paths = {}
        for relative, document in zip(_workspace_paths(documents), documents, strict=True):
            target = corpus / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(document.text, encoding="utf-8")
            self._paths[relative] = document.doc_id
        self._corpus = corpus

    def _call(self, question: str, top_k: int) -> ExternalReply:
        if self._corpus is None:
            raise RuntimeError(f"{self.name}: ingest the documents before asking questions")
        workspace = Path(tempfile.mkdtemp(prefix="ragbench-agent-"))
        try:
            prompt = self._prepare(workspace, question)
            with self._limiter.acquire() if self._limiter is not None else nullcontext():
                trajectory, stopped = self._run(self._argv(prompt, workspace), workspace)
            answer = self._answer(workspace, trajectory)
        finally:
            _remove(workspace)
        return self._reply(trajectory, answer, stopped, workspace)

    # -- the workspace ------------------------------------------------------------------------------------------------------------------

    def _prepare(self, workspace: Path, question: str) -> str:
        assert self._corpus is not None
        docs = workspace / DOCS_DIR
        shutil.copytree(self._corpus, docs)
        _read_only(docs)
        answer_file = self.options.answer_file
        deliver = f"Write your final answer to the file `{answer_file}`." if answer_file else "Your final message is your answer."
        prompt = TASK.format(deliver=deliver, question=question.strip())
        (workspace / TASK_FILE).write_text(prompt, encoding="utf-8")
        instructions = self._instruction_text()
        if instructions and self.options.preset != "claude_code":
            (workspace / INSTRUCTIONS_FILE).write_text(instructions, encoding="utf-8")
        if self.options.preset == "opencode" and not self.force_mock:
            (workspace / "opencode.json").write_text(_opencode_config(self.options.network), encoding="utf-8")
        for target, source in self._files.items():
            destination = workspace / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(source, destination)
        return prompt

    def _instruction_text(self) -> str:
        return "\n\n".join(part.strip() for part in (self.options.system_prompt, self._instructions) if part and part.strip())

    def _argv(self, prompt: str, workspace: Path) -> list[str]:
        opts = self.options
        if self.force_mock:  # `--mock` never starts a harness, whatever it is
            return [sys.executable, "-m", "ragbench.rag_systems.cli_agent_mock", "--format", self.output_format, *(["--answer-file", opts.answer_file] if opts.answer_file else [])]
        if opts.command:
            values = {"{prompt}": prompt, "{model}": self.harness_model or "", "{workspace}": str(workspace), "{task_file}": str(workspace / TASK_FILE)}
            return [_fill(argument, values) for argument in opts.command]
        model = self.harness_model
        if opts.preset == "claude_code":
            tools = "Read,Grep,Glob" + (",WebFetch,WebSearch" if opts.network == "on" else "")
            # `--restricted` confines the file tools to the workspace and ignores your settings files and plugins (`--bare` would also drop Grep
            # and Glob); `--strict-mcp-config` skips your MCP servers and `--disable-slash-commands` your skills.
            argv = ["claude", "-p", prompt, "--output-format", "stream-json", "--verbose", "--restricted", "--strict-mcp-config", "--disable-slash-commands"]
            argv += ["--no-session-persistence", "--tools", tools]
            argv += ["--model", model] if model else []
            if instructions := self._instruction_text():
                argv += ["--append-system-prompt", instructions]
            return argv
        if opts.preset == "codex":
            sandbox = ["--sandbox", "read-only"] if opts.network == "off" else ["--sandbox", "workspace-write", "-c", "sandbox_workspace_write.network_access=true"]
            # `--ignore-user-config`: not your `config.toml` (its model, reasoning effort, plugins); Codex's own defaults, as installed. Auth is still yours.
            return [
                "codex", "exec", "--json", "--skip-git-repo-check", "--ephemeral", "--ignore-user-config", "--color", "never", *sandbox,
                "-C", str(workspace), *(["-m", model] if model else []), "--", prompt,
            ]  # fmt: skip
        return ["opencode", "run", "--format", "json", "--pure", "--dir", str(workspace), *(["-m", model] if model else []), prompt]

    # -- running the harness ------------------------------------------------------------------------------------------------------------

    def _run(self, argv: list[str], workspace: Path) -> tuple[Trajectory, bool]:
        """Run the harness, feeding its stdout to the parser line by line. Returns the trajectory and whether the step cap stopped it."""
        parser = PARSERS[self.output_format]()
        # `PWD` is the workspace: a harness that trusts it over its real working directory (OpenCode does) would otherwise work in yours.
        env = {**os.environ, "PWD": str(workspace), **{name: fill_env(value) for name, value in self.options.env.items()}}
        env.pop("OLDPWD", None)
        timed_out = threading.Event()
        stopped = False
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as stderr:
            try:
                process = subprocess.Popen(
                    argv, cwd=workspace, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=stderr,
                    text=True, encoding="utf-8", errors="replace", start_new_session=True,
                )  # fmt: skip
            except FileNotFoundError:
                raise RuntimeError(f"{self.name}: `{argv[0]}` is not installed or not on PATH (preset {self.options.preset})") from None

            def expire() -> None:
                timed_out.set()
                _stop(process)

            timer = threading.Timer(self.options.timeout_s, expire)
            timer.start()
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    parser.feed(line)
                    if len(parser.trajectory.tool_calls) > self.options.max_steps:
                        stopped = True
                        _stop(process)
                        break
            finally:
                timer.cancel()
                _drain(process.stdout)
                process.wait()
            trajectory = parser.trajectory
            if timed_out.is_set() or (not stopped and (trajectory.error or process.returncode != 0)):
                self._charge_partial(trajectory)  # the harness was billed for the turns it ran, even though the answer is lost
            if timed_out.is_set():
                raise RuntimeError(f"{self.name}: the harness did not finish within {self.options.timeout_s:g} s")
            if stopped:
                calls = trajectory.tool_calls[self.options.max_steps :]
                trajectory.items = [item for item in trajectory.items if not any(item is call for call in calls)]
            elif trajectory.error:
                raise RuntimeError(f"{self.name}: the harness reported an error: {truncate(trajectory.error, STDERR_TAIL)}")
            elif process.returncode != 0:
                stderr.seek(0)
                tail = stderr.read()[-STDERR_TAIL:].strip()
                raise RuntimeError(f"{self.name}: `{argv[0]}` exited with code {process.returncode}" + (f": {tail}" if tail else ""))
        return trajectory, stopped

    def _charge_partial(self, trajectory: Trajectory) -> None:
        costs, known = self._turn_costs(trajectory)
        if costs:
            note_spend(sum_costs(costs.values()))
        elif known and trajectory.cost_usd:  # a reported total and no turns to spread it over
            note_spend(CostBreakdown(llm_cost=trajectory.cost_usd))

    def _answer(self, workspace: Path, trajectory: Trajectory) -> str:
        if self.options.answer_file:
            path = workspace / self.options.answer_file
            if path.is_file() and (text := path.read_text(encoding="utf-8", errors="replace").strip()):
                return text
        return trajectory.final_text().strip()

    # -- the trajectory as steps, passages and cost -------------------------------------------------------------------------------------

    def _reply(self, trajectory: Trajectory, answer: str, stopped: bool, workspace: Path) -> ExternalReply:
        turns = trajectory.turns
        costs, cost_known = self._turn_costs(trajectory)
        last_turn = turns[-1] if turns else None
        steps: list[Step] = []
        contexts: dict[str, dict[str, Any]] = {}
        for item in trajectory.items:
            if isinstance(item, Turn):
                final = item is last_turn and not stopped
                steps.append(
                    Step(
                        kind="generate" if final else "llm",
                        name="answer" if final else "turn",
                        cost=costs[id(item)],
                        prompt_tokens=item.prompt_tokens,
                        completion_tokens=item.completion_tokens,
                        output_preview=truncate(item.text, PREVIEW_CHARS) if item.text else None,
                    )
                )
            else:
                steps.append(self._tool_step(item, contexts, workspace))
        if not turns and trajectory.events and cost_known and trajectory.cost_usd:
            steps.append(Step(kind="generate", name="answer", cost=CostBreakdown(llm_cost=trajectory.cost_usd), output_preview=truncate(answer, PREVIEW_CHARS)))
        prompt = sum(turn.prompt_tokens for turn in turns)
        completion = sum(turn.completion_tokens for turn in turns)
        agent = {
            "harness": self.options.preset,
            "steps": len(turns),
            "llm_calls": len(turns),
            "tool_calls": len(trajectory.tool_calls),
            "budget_exhausted": stopped,
            "termination": "max_steps" if stopped else "finished",
            "max_prompt_tokens": max((turn.prompt_tokens for turn in turns), default=0),
            "network": self.network,
            "outside_workspace_calls": sum(1 for step in steps if step.metadata.get("outside_workspace")),
        }
        if agent["outside_workspace_calls"]:
            logger.warning("%s: the harness reached files outside its workspace (%d calls); see the steps' `outside_workspace`", self.name, agent["outside_workspace_calls"])
        return ExternalReply(
            answer=answer,
            contexts=_ranked(contexts) if trajectory.events else None,
            prompt_tokens=prompt,
            completion_tokens=completion,
            cost=sum_costs(step.cost for step in steps) if cost_known else None,
            steps=steps,
            agent=agent,
        )

    def _tool_step(self, call: ToolCall, contexts: dict[str, dict[str, Any]], workspace: Path) -> Step:
        """A tool call as a `tool` step; the documents it read (their text as the agent saw it) or found (the matching lines) become passages."""
        kind = action_kind(call)
        doc_ids = []
        for path in evidence_paths(call):
            doc_id = self._paths.get(path)
            if doc_id is None:
                continue
            doc_ids.append(doc_id)
            if kind == "read":
                seen = contexts.get(doc_id)
                text = call.output if seen is None or not seen.get("read") else f"{seen['text']}\n{call.output}"
                order = seen["read"] if seen is not None and seen.get("read") else 1 + sum(1 for c in contexts.values() if c.get("read"))
                contexts[doc_id] = {"doc_id": doc_id, "text": text[:MAX_CONTEXT_CHARS], "read": order}  # `read`: its rank among the files read
            elif doc_id not in contexts:
                lines = [line for line in call.output.splitlines() if f"{DOCS_DIR}/{path}" in line]
                contexts[doc_id] = {"doc_id": doc_id, "text": "\n".join(lines)[:MAX_CONTEXT_CHARS] or call.output[:MAX_CONTEXT_CHARS]}
        metadata: dict[str, Any] = {"action": kind, "doc_ids": list(dict.fromkeys(doc_ids))}
        if call.error:
            metadata["error"] = call.error
        if outside := _outside(call, workspace):
            metadata["outside_workspace"] = outside
        return Step(
            kind="tool",
            name=call.name,
            input_preview=truncate(input_text(call), PREVIEW_CHARS) or None,
            output_preview=truncate(call.output, PREVIEW_CHARS) or None,
            metadata=metadata,
        )

    def _turn_costs(self, trajectory: Trajectory) -> tuple[dict[int, CostBreakdown], bool]:
        """Each turn's cost, and whether the answer's cost is known: the turns' own reported costs, else the reported total spread over the
        turns by tokens, else their tokens at the declared model's price, else unknown ($0 with the tokens kept)."""
        turns = trajectory.turns
        model = self.options.model
        priced = bool(model) and pricing.price_known(str(model))
        own = bool(turns) and all(turn.cost_usd is not None for turn in turns)
        tokens = [turn.prompt_tokens + turn.completion_tokens for turn in turns]
        costs: dict[int, CostBreakdown] = {}
        for index, turn in enumerate(turns):
            if own:
                usd = float(turn.cost_usd or 0.0)
            elif trajectory.cost_usd is not None:
                share = tokens[index] / sum(tokens) if sum(tokens) else float(index == len(turns) - 1)
                usd = trajectory.cost_usd * share
            elif priced:
                usd = pricing.estimate_model_cost(str(model), turn.prompt_tokens, turn.completion_tokens)
            else:
                usd = 0.0
            costs[id(turn)] = CostBreakdown(llm_prompt_tokens=turn.prompt_tokens, llm_completion_tokens=turn.completion_tokens, llm_cost=usd)
        known = own or trajectory.cost_usd is not None or (priced and bool(turns))
        return costs, known


def global_instructions(preset: str) -> list[Path]:
    """The instruction files of yours a preset's harness may read on every run whatever the workspace holds, when they have content."""
    home = Path.home()
    config = Path(os.environ.get("XDG_CONFIG_HOME") or home / ".config")
    candidates = {
        "claude_code": [home / ".claude" / "CLAUDE.md"],
        "codex": [Path(os.environ.get("CODEX_HOME") or home / ".codex") / "AGENTS.md"],
        "opencode": [config / "opencode" / "AGENTS.md"],
    }.get(preset, [])
    return [path for path in candidates if path.is_file() and path.read_text(encoding="utf-8", errors="replace").strip()]


def _ranked(contexts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """The passages, ranked: the files the agent read, in the order it first read them, then the files it only saw in search results."""
    ordered = sorted((c for c in contexts.values() if c.get("read")), key=lambda c: c["read"]) + [c for c in contexts.values() if not c.get("read")]
    return [{"doc_id": c["doc_id"], "text": c["text"]} for c in ordered]


def _outside(call: ToolCall, workspace: Path) -> list[str]:
    """The paths outside the workspace that a file tool was pointed at (a leak tripwire: the harness should see nothing but its workspace).
    Shell commands are not checked: Codex's sandbox and OpenCode's config deny the ways out instead."""
    if not isinstance(call.input, dict):
        return []
    root = workspace.resolve()
    found = []
    for key in PATH_KEYS:
        value = call.input.get(key)
        if isinstance(value, str) and value.startswith("/") and not Path(value).resolve().is_relative_to(root):
            found.append(value)
    return found


def _harness_model(preset: str, model: str | None) -> str | None:
    """The harness's spelling of a model ref: the bare name, `provider/name` for OpenCode, the ref as written for a custom command."""
    if not model:
        return None
    if preset == "custom":
        return model
    provider, name = parse_model_ref(model)
    return f"{provider}/{name}" if preset == "opencode" else name


def _fill(argument: str, values: dict[str, str]) -> str:
    for placeholder, value in values.items():
        argument = argument.replace(placeholder, value)
    return argument


def _opencode_config(network: str) -> str:
    import json

    permission = {"edit": "deny", "bash": "deny", "webfetch": "allow" if network == "on" else "deny"}
    return json.dumps({"$schema": "https://opencode.ai/config.json", "permission": permission}, indent=2)


def _workspace_paths(documents: list[Document]) -> list[str]:
    """Where each document goes under `docs/`: its path relative to the corpus's common directory, so that the file names the agent sees are
    the corpus's own (`docs/policies/refunds.md`). A file that is not plain text was converted, so it gets `.txt` (`report.pdf.txt`); a path
    that would clash or escape becomes `<doc_id>/<file name>`."""
    paths = [PurePosixPath(Path(document.path).as_posix()) for document in documents]
    common = None
    if paths and len({path.is_absolute() for path in paths}) == 1:
        common = PurePosixPath(os.path.commonpath([str(path.parent) for path in paths]))
    taken: set[str] = set()
    result = []
    for path, document in zip(paths, documents, strict=True):
        relative = path.relative_to(common) if common is not None else path
        parts = [part for part in relative.parts if part not in ("", ".", "..", "/")]
        candidate = "/".join(parts) if parts and not relative.is_absolute() else ""
        if not candidate or candidate in taken:
            candidate = f"{document.doc_id}/{path.name or document.doc_id}"
        if PurePosixPath(candidate).suffix.lower() not in TEXT_SUFFIXES:
            candidate += ".txt"
        taken.add(candidate)
        result.append(candidate)
    return result


def _read_only(root: Path) -> None:
    for path in root.rglob("*"):
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH | (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH if path.is_dir() else 0))
    root.chmod(0o555)


def _remove(root: Path) -> None:
    """Delete a workspace, its read-only parts included."""
    if not root.exists():
        return
    for path in [root, *root.rglob("*")]:
        if path.is_dir() and not path.is_symlink():
            path.chmod(0o755)
    shutil.rmtree(root, ignore_errors=True)


def _stop(process: subprocess.Popen) -> None:
    """Stop the harness and everything it started."""
    try:
        if hasattr(os, "killpg"):
            os.killpg(process.pid, signal.SIGTERM)
        else:  # pragma: no cover - Windows
            process.terminate()
        process.wait(timeout=5)
    except (ProcessLookupError, PermissionError):
        return
    except subprocess.TimeoutExpired:
        if hasattr(os, "killpg"):
            os.killpg(process.pid, signal.SIGKILL)
        else:  # pragma: no cover - Windows
            process.kill()


def _drain(stream: IO[str] | None) -> None:
    if stream is not None:
        try:
            stream.read()
        except (OSError, ValueError):
            pass
        stream.close()
