"""`cli_agent`: agent harnesses (Claude Code, Codex, OpenCode, any command) as systems, and the trajectory metrics of every agentic system.

No real harness runs here: `tests/fake_cli_agent.py` replays recorded event streams (`tests/fixtures/cli_agent/`), and `--mock` runs the
bundled offline harness (`ragbench.rag_systems.cli_agent_mock`)."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml
from dataset_support import DOCUMENTS, QUESTIONS

from ragbench.config.schema import SystemConfig
from ragbench.documents.schema import Document
from ragbench.evaluation.evaluator import run_benchmark
from ragbench.evaluation.trajectory_metrics import redundant, steps_to_evidence, summary_fields
from ragbench.rag_systems import create_rag_system
from ragbench.rag_systems.cli_agent_rag import _workspace_paths
from ragbench.rag_systems.cli_trajectory import ToolCall, action_kind, evidence_paths, parse
from ragbench.utils.jsonl import read_jsonl

TESTS = Path(__file__).resolve().parent
FAKE = str(TESTS / "fake_cli_agent.py")
FIXTURES = TESTS / "fixtures" / "cli_agent"
QUESTION = "How much does HarborShield cost per month?"


def _documents() -> list[Document]:
    return [Document(doc_id=doc_id, path=f"corpus/{doc_id}.md", title=doc_id, text=text) for doc_id, text in DOCUMENTS.items()]


def _fixture(name: str, workspace: str = "/ws") -> str:
    return (FIXTURES / f"{name}.jsonl").read_text(encoding="utf-8").replace("{workspace}", workspace)


def _agent(fmt: str = "claude_code", *args: str, force_mock: bool = False, **options):
    """A `custom` cli_agent running the fake harness on the recorded stream of `fmt`."""
    command = [sys.executable, FAKE, str(FIXTURES / f"{fmt}.jsonl"), *args]
    retrieval = {"preset": "custom", "command": command, "output_format": fmt, "network": "on", **options}
    system = create_rag_system(SystemConfig(type="cli_agent", name="agent", retrieval=retrieval), force_mock=force_mock)
    system.ingest(_documents())
    return system


# -- parsers -----------------------------------------------------------------------------------------------------------------------------


def test_claude_code_stream_json_becomes_turns_tool_calls_answer_and_cost():
    trajectory = parse("claude_code", _fixture("claude_code"))

    kinds = [type(item).__name__ for item in trajectory.items]
    assert kinds == ["Turn", "ToolCall", "Turn", "ToolCall", "Turn", "ToolCall", "ToolCall", "Turn"]
    first = trajectory.turns[0]
    assert (first.prompt_tokens, first.completion_tokens) == (3184, 71)  # one message in two events: usage counted once
    assert [call.name for call in trajectory.tool_calls] == ["Grep", "Read", "Read", "Grep"]
    assert trajectory.tool_calls[2].error and "does not exist" in trajectory.tool_calls[2].error
    assert "$200 per month" in trajectory.tool_calls[1].output
    assert trajectory.answer.startswith("HarborShield costs $200") and trajectory.cost_usd == pytest.approx(0.0012437)
    assert trajectory.model == "claude-haiku-5-5" and trajectory.error is None


def test_codex_jsonl_becomes_shell_calls_one_turn_and_the_last_message():
    trajectory = parse("codex", _fixture("codex"))

    assert [call.name for call in trajectory.tool_calls] == ["shell"] * 5  # started + completed is one call; reasoning is not an action
    assert [call.error for call in trajectory.tool_calls] == [None, None, None, "exit code 1", None]  # rg finding nothing is not an error
    assert [action_kind(call) for call in trajectory.tool_calls] == ["search", "read", "search", "read", "search"]
    (turn,) = trajectory.turns
    assert (turn.prompt_tokens, turn.completion_tokens) == (24310, 412) and trajectory.cost_usd is None
    assert "$200 per month" in trajectory.final_text() and isinstance(trajectory.items[-1], type(turn))


def test_opencode_json_becomes_steps_with_their_tokens_and_costs():
    trajectory = parse("opencode", _fixture("opencode"))

    assert [call.name for call in trajectory.tool_calls] == ["grep", "read", "webfetch"]
    assert trajectory.tool_calls[2].error and "rule" in trajectory.tool_calls[2].error
    assert [(t.prompt_tokens, t.completion_tokens) for t in trajectory.turns] == [(2950, 165), (6300, 140), (3920, 70)]
    assert trajectory.cost_usd == pytest.approx(0.00074)
    assert trajectory.final_text() == "HarborShield costs $200 per month for the marine module (docs/doc_001.md)."


def test_opencode_prices_its_tokens_reasoning_included_at_the_cheapest_openrouter_list_price():
    # OpenCode 1.19.4's step cost is its tokens at its own price table, which for `openrouter/z-ai/glm-5.3` held the cheapest upstream's list
    # price ($0.079 input, $1.40 output and reasoning, $0.078 cache read per 1M tokens) when this was recorded. OpenRouter bills the upstream
    # it routes to, so this cost is a floor; with an `openrouter:` model the `opencode` preset records OpenRouter's bill instead.
    finishes = [json.loads(line)["part"] for line in _fixture("opencode_recorded").splitlines() if '"step_finish"' in line]
    for part in finishes:
        tokens = part["tokens"]
        list_price = (tokens["input"] * 0.079 + (tokens["output"] + tokens["reasoning"]) * 1.40 + tokens["cache"]["read"] * 0.078) / 1e6
        assert part["cost"] == pytest.approx(list_price, rel=1e-6)
    trajectory = parse("opencode", _fixture("opencode_recorded"))
    assert [t.completion_tokens for t in trajectory.turns] == [f["tokens"]["output"] + f["tokens"]["reasoning"] for f in finishes]
    assert trajectory.cost_usd == pytest.approx(0.00658448)


def test_text_output_is_the_answer_and_no_trajectory():
    trajectory = parse("text", "banner\nThe answer is $200.\n")
    assert trajectory.answer == "banner\nThe answer is $200." and trajectory.items == [] and trajectory.events is False


def test_non_json_lines_and_unknown_events_are_skipped():
    trajectory = parse("claude_code", 'warning: something\n{"type":"rate_limit_event"}\n[1,2]\n' + _fixture("claude_code"))
    assert len(trajectory.tool_calls) == 4


def test_a_cost_claude_code_guessed_is_dropped_and_the_final_usage_is_booked_on_the_last_turn():
    """Claude Code prices a model it does not know on a guess (`costBasis: unknown`), and its stream under-reports each message's output."""
    stream = "\n".join([
        '{"type":"assistant","message":{"id":"m1","content":[{"type":"tool_use","id":"t1","name":"Read","input":{"file_path":"docs/a.md"}}],"usage":{"input_tokens":4,"cache_read_input_tokens":800,"output_tokens":7}}}',
        '{"type":"assistant","message":{"id":"m2","content":[{"type":"text","text":"done"}],"usage":{"input_tokens":2,"cache_creation_input_tokens":300,"cache_read_input_tokens":804,"output_tokens":5}}}',
        '{"type":"result","subtype":"success","is_error":false,"result":"done","total_cost_usd":0.0359,"usage":{"input_tokens":6,"cache_creation_input_tokens":300,"cache_read_input_tokens":1604,"output_tokens":1238},'
        '"modelUsage":{"claude-haiku-5-5":{"costUSD":0.0359,"costBasis":"unknown"}}}',
    ])  # fmt: skip
    trajectory = parse("claude_code", stream)
    assert trajectory.cost_usd is None
    assert [(t.prompt_tokens, t.completion_tokens) for t in trajectory.turns] == [(804, 7), (1106, 1231)]


def test_a_file_tool_pointed_outside_the_workspace_is_flagged(tmp_path, caplog):
    stream = '{"type":"assistant","message":{"id":"m1","content":[{"type":"tool_use","id":"t1","name":"Read","input":{"file_path":"/etc/hosts"}},' \
             '{"type":"tool_use","id":"t2","name":"Read","input":{"file_path":"{workspace}/docs/doc_001.md"}}],"usage":{"input_tokens":5,"output_tokens":5}}}\n'
    (tmp_path / "leak.jsonl").write_text(stream + '{"type":"result","subtype":"success","result":"x","total_cost_usd":0.001}\n', encoding="utf-8")
    command = [sys.executable, FAKE, str(tmp_path / "leak.jsonl")]
    system = create_rag_system(SystemConfig(type="cli_agent", name="leaky", retrieval={"preset": "custom", "command": command, "output_format": "claude_code", "network": "on"}))
    system.ingest(_documents())
    with caplog.at_level(logging.WARNING):
        result = system.answer_question(QUESTION)
    tools = [step for step in result.steps if step.kind == "tool"]
    assert tools[0].metadata["outside_workspace"] == ["/etc/hosts"] and "outside_workspace" not in tools[1].metadata
    assert result.metadata["agent"]["outside_workspace_calls"] == 1 and "outside its workspace" in caplog.text


def test_a_harness_error_result_is_recorded():
    trajectory = parse("claude_code", '{"type":"result","subtype":"error_max_turns","is_error":true,"total_cost_usd":0.01}\n')
    assert trajectory.error and trajectory.error.startswith("error_max_turns")


def test_actions_are_classified_and_only_shown_documents_count_as_evidence():
    read = ToolCall("Read", {"file_path": "/tmp/ws/docs/sub/a.md"})
    grep = ToolCall("Grep", {"pattern": "x"}, output="Found 2 files\n/tmp/ws/docs/a.md\ndocs/b.md:3:text")
    glob = ToolCall("Glob", {"pattern": "**/*.md"}, output="/tmp/ws/docs/a.md\n/tmp/ws/docs/b.md")
    shell_read = ToolCall("shell", "bash -lc \"sed -n '1,80p' docs/c.md\"")
    shell_list = ToolCall("shell", "bash -lc 'ls -la docs'", output="docs/a.md")
    piped = ToolCall("shell", "/bin/zsh -lc 'cd docs && LC_ALL=C grep -rn tenant .'", output="docs/d.md:1:tenant")
    assert [action_kind(c) for c in (read, grep, glob, shell_read, shell_list, piped)] == ["read", "search", "list", "read", "list", "search"]
    assert evidence_paths(read) == ["sub/a.md"] and evidence_paths(grep) == ["b.md"] and evidence_paths(glob) == []  # a bare file list shows no text
    assert action_kind(ToolCall("shell", "/bin/bash -lc 'rg --files docs'", output="docs/a.md")) == "list"
    opencode_grep = ToolCall("grep", {"pattern": "x"}, output="Found 2 matches\n/ws/docs/a.md:\n  Line 5: HarborShield\n\n/ws/docs/b.md:")
    assert evidence_paths(opencode_grep) == ["a.md"]
    assert evidence_paths(shell_read) == ["c.md"] and evidence_paths(piped) == ["d.md"]
    assert evidence_paths(ToolCall("fetch_doc", {"path": "mydocs/x.md"}, output="see docs/e.md.")) == ["e.md"]


# -- the system ----------------------------------------------------------------------------------------------------------------------------


def test_a_recorded_claude_code_run_becomes_steps_passages_cost_and_an_agent_block():
    result = _agent("claude_code").answer_question(QUESTION, top_k=5)

    assert result.answer.startswith("HarborShield costs $200")
    assert [(step.kind, step.name) for step in result.steps] == [
        ("llm", "turn"), ("tool", "Grep"), ("llm", "turn"), ("tool", "Read"), ("llm", "turn"), ("tool", "Read"), ("tool", "Grep"), ("generate", "answer"),
    ]  # fmt: skip
    grep, read, missing = result.steps[1], result.steps[3], result.steps[5]
    assert grep.metadata == {"action": "search", "doc_ids": []} and read.metadata == {"action": "read", "doc_ids": ["doc_001"]}  # files_with_matches shows no text
    assert missing.metadata["doc_ids"] == [] and "does not exist" in missing.metadata["error"]
    chunks = result.retrieval_result.chunks
    assert [chunk.doc_id for chunk in chunks] == ["doc_001"]
    assert "$950 per month" in chunks[0].text
    assert result.cost.total_cost == pytest.approx(0.0012437) and sum(s.cost.total_cost for s in result.steps) == pytest.approx(0.0012437)
    assert result.token_usage == {"prompt_tokens": 3184 + 3326 + 3536 + 3631, "completion_tokens": 71 + 58 + 90 + 24}
    assert result.metadata["agent"] == {
        "harness": "custom", "steps": 4, "llm_calls": 4, "tool_calls": 4, "budget_exhausted": False, "termination": "finished",
        "max_prompt_tokens": 3631, "network": "on", "outside_workspace_calls": 0,
    }  # fmt: skip
    assert result.metadata["external"] == {"unmapped_doc_ids": [], "cost_known": True}


def test_codex_tokens_are_priced_at_the_declared_model_and_an_unpriced_model_is_unknown_cost():
    priced = _agent("codex", model="gpt-6-luna").answer_question(QUESTION)
    assert priced.cost.llm_cost > 0 and priced.metadata["external"]["cost_known"] is True
    assert [chunk.doc_id for chunk in priced.retrieval_result.chunks] == ["doc_001", "doc_005"]
    unknown = _agent("codex").answer_question(QUESTION)
    assert unknown.cost.total_cost == 0 and unknown.metadata["external"]["cost_known"] is False and unknown.token_usage["prompt_tokens"] == 24310


def test_opencode_costs_are_the_reported_per_step_costs():
    result = _agent("opencode").answer_question(QUESTION)
    assert [step.cost.llm_cost for step in result.steps if step.kind in ("llm", "generate")] == pytest.approx([0.00021, 0.00034, 0.00019])
    assert result.metadata["external"]["cost_known"] is True and result.steps[-1].kind == "generate"


def test_a_text_harness_has_no_passages_and_one_external_step(tmp_path):
    (tmp_path / "answer.txt").write_text("It costs $200.\n", encoding="utf-8")
    system = create_rag_system(SystemConfig(type="cli_agent", name="plain", retrieval={"preset": "custom", "command": ["cat", str(tmp_path / "answer.txt")], "network": "on"}))
    system.ingest(_documents())
    result = system.answer_question(QUESTION)
    assert result.answer == "It costs $200." and [step.kind for step in result.steps] == ["external"]
    assert result.metadata["external"]["no_contexts"] is True and result.retrieval_result.chunks == []


def test_the_workspace_holds_a_read_only_corpus_and_the_task_and_never_the_references(tmp_path):
    dump = tmp_path / "dump.json"
    instructions = tmp_path / "house_rules.md"
    instructions.write_text("Cite the file name.", encoding="utf-8")
    skills = tmp_path / "skills"
    (skills / "lookup").mkdir(parents=True)
    (skills / "lookup" / "SKILL.md").write_text("# Lookup skill", encoding="utf-8")
    system = _agent("claude_code", "--dump", str(dump), system_prompt="Be brief.", instructions=str(instructions), files={".agent/skills": str(skills)}, env={"MY_FLAG": "${HOME}"})
    system.answer_question(QUESTION)

    seen = json.loads(dump.read_text(encoding="utf-8"))
    assert set(seen["files"]) == {*(f"docs/{doc_id}.md" for doc_id in DOCUMENTS), "TASK.md", "AGENTS.md", ".agent/skills/lookup/SKILL.md"}
    assert seen["files"]["docs/doc_001.md"] == DOCUMENTS["doc_001"] and seen["docs_writable"] is False
    assert QUESTION in seen["files"]["TASK.md"] and "docs/" in seen["files"]["TASK.md"]
    assert seen["files"]["AGENTS.md"] == "Be brief.\n\nCite the file name."
    assert seen["env"]["MY_FLAG"] == str(Path.home())
    assert Path(seen["env"]["PWD"]).resolve() == Path(seen["cwd"]).resolve() and "OLDPWD" not in seen["env"]  # never the caller's directory
    everything = json.dumps(seen["files"])
    for question in QUESTIONS:  # no reference answer and no label reaches the harness
        assert question["reference_answer"] not in everything.replace(DOCUMENTS.get(question.get("relevant_doc_ids", [""])[0], ""), "")
    assert "relevant_doc_ids" not in everything and "reference_answer" not in everything
    assert not Path(seen["cwd"]).exists()  # removed after the question


def test_the_answer_file_wins_over_the_final_message():
    result = _agent("claude_code", "--answer-file", "ANSWER.md", "--answer", "Exactly $200.", answer_file="ANSWER.md").answer_question(QUESTION)
    assert result.answer == "Exactly $200."


def test_the_step_cap_stops_the_harness_and_keeps_what_it_did():
    result = _agent("claude_code", "--repeat-tool", "6", "--sleep", "2", max_steps=3).answer_question(QUESTION)

    assert [step.kind for step in result.steps].count("tool") == 3 and result.steps[-1].kind == "llm"  # no answer turn: it was stopped
    assert result.metadata["agent"]["budget_exhausted"] is True and result.metadata["agent"]["termination"] == "max_steps"
    assert result.latency_ms < 2000  # stopped, not waited for


def test_a_slow_harness_times_out(tmp_path):
    with pytest.raises(RuntimeError, match="did not finish within 0.5 s"):
        _agent("claude_code", "--sleep", "5", timeout_s=0.5).answer_question(QUESTION)


def test_a_failing_harness_fails_the_question_with_its_stderr():
    with pytest.raises(RuntimeError, match="exited with code 3: boom: no credentials"):
        _agent("claude_code", "--exit-code", "3", "--stderr", "boom: no credentials", output_format="text").answer_question(QUESTION)
    with pytest.raises(RuntimeError, match="reported an error: error_during_execution"):
        system = create_rag_system(SystemConfig(type="cli_agent", name="x", retrieval={"preset": "custom", "network": "on", "output_format": "claude_code",
                                                                                         "command": ["echo", '{"type":"result","subtype":"error_during_execution","is_error":true}']}))
        system.ingest(_documents())
        system.answer_question(QUESTION)


def test_a_missing_harness_says_it_is_not_installed(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    system = create_rag_system(SystemConfig(type="cli_agent", name="cc", retrieval={"preset": "claude_code"}))
    system.ingest(_documents())
    with pytest.raises(RuntimeError, match="`claude` is not installed"):
        system.answer_question(QUESTION)


@pytest.mark.parametrize("fmt", ["claude_code", "codex", "opencode", "text"])
def test_mock_never_starts_the_harness_and_runs_the_offline_one_in_its_format(fmt, monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")  # neither `claude` nor anything else could start
    preset = fmt if fmt != "text" else "custom"
    retrieval = {"preset": preset, "model": "anthropic:claude-haiku-5-5"} | ({"command": ["my-harness"]} if preset == "custom" else {})
    system = create_rag_system(SystemConfig(type="cli_agent", name="m", retrieval=retrieval), force_mock=True)
    system.ingest(_documents())
    result = system.answer_question(QUESTION)

    assert "$200" in result.answer
    if fmt == "text":
        assert result.metadata["external"]["no_contexts"] is True
    else:
        assert result.retrieval_result.chunks[0].doc_id == "doc_001" and any(step.kind == "tool" for step in result.steps)
        assert result.metadata["agent"]["tool_calls"] >= 2


def test_presets_build_the_harness_command_with_network_tools_off(tmp_path):
    def argv(preset, **options):
        system = create_rag_system(SystemConfig(type="cli_agent", name=preset, retrieval={"preset": preset, **options}))
        return system._argv("PROMPT", tmp_path)

    claude = argv("claude_code", model="anthropic:claude-haiku-5-5", system_prompt="Be brief.")
    assert claude[:3] == ["claude", "-p", "PROMPT"] and claude[claude.index("--tools") + 1] == "Read,Grep,Glob"
    assert {"--restricted", "--strict-mcp-config", "--disable-slash-commands"} <= set(claude) and "--bare" not in claude  # --bare has no Grep/Glob
    assert claude[claude.index("--model") + 1] == "claude-haiku-5-5" and claude[claude.index("--append-system-prompt") + 1] == "Be brief."
    assert "--no-session-persistence" in claude and "stream-json" in claude
    assert "WebFetch" in argv("claude_code", network="on")[argv("claude_code", network="on").index("--tools") + 1]
    codex = argv("codex", model="gpt-6-luna")
    assert codex[:3] == ["codex", "exec", "--json"] and codex[codex.index("--sandbox") + 1] == "read-only" and codex[-2:] == ["--", "PROMPT"]
    assert "--ignore-user-config" in codex  # not your config.toml: its model, reasoning effort and plugins would change what is measured
    assert codex[codex.index("-m") + 1] == "gpt-6-luna" and codex[codex.index("-C") + 1] == str(tmp_path)
    assert "sandbox_workspace_write.network_access=true" in argv("codex", network="on")
    opencode = argv("opencode", model="openrouter:z-ai/glm-5.3")
    assert opencode == ["opencode", "run", "--format", "json", "--pure", "--dir", str(tmp_path), "-m", "openrouter/z-ai/glm-5.3", "PROMPT"]
    assert argv("opencode", model="x", harness_model="openrouter/qwen/qwen3.8-flash")[8] == "openrouter/qwen/qwen3.8-flash"
    custom = argv("custom", command=["my-agent", "--model={model}", "--cwd", "{workspace}", "{task_file}", "{prompt}"], model="gpt-6-luna", network="on")
    assert custom == ["my-agent", "--model=gpt-6-luna", "--cwd", str(tmp_path), str(tmp_path / "TASK.md"), "PROMPT"]


def test_opencode_gets_a_config_that_denies_web_shell_and_edits(tmp_path):
    system = create_rag_system(SystemConfig(type="cli_agent", name="oc", retrieval={"preset": "opencode"}))
    system.ingest(_documents())
    system._prepare(tmp_path, QUESTION)
    config = json.loads((tmp_path / "opencode.json").read_text(encoding="utf-8"))
    assert config["permission"] == {"edit": "deny", "bash": "deny", "webfetch": "deny"} and "provider" not in config  # no relay off OpenRouter


def test_a_custom_harness_with_network_off_is_warned_about(caplog):
    with caplog.at_level(logging.WARNING):
        system = create_rag_system(SystemConfig(type="cli_agent", name="mine", retrieval={"preset": "custom", "command": ["x"]}))
    assert "cannot switch a custom harness's network off" in caplog.text and system.network == "unenforced"


def test_a_non_empty_global_instructions_file_is_warned_about(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    (tmp_path / "codex").mkdir()
    (tmp_path / "codex" / "AGENTS.md").write_text("", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        create_rag_system(SystemConfig(type="cli_agent", name="cx", retrieval={"preset": "codex"}))
    assert "AGENTS.md" not in caplog.text  # empty: nothing reaches the prompt
    (tmp_path / "codex" / "AGENTS.md").write_text("Always answer in French.", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        create_rag_system(SystemConfig(type="cli_agent", name="cx", retrieval={"preset": "codex"}))
        create_rag_system(SystemConfig(type="cli_agent", name="mocked", retrieval={"preset": "codex"}), force_mock=True)
    assert caplog.text.count("AGENTS.md is not empty") == 1  # and never under --mock, which starts no harness


def test_options_are_checked(tmp_path):
    with pytest.raises(ValueError, match="needs a `command`"):
        create_rag_system(SystemConfig(type="cli_agent", retrieval={"preset": "custom"}))
    with pytest.raises(ValueError, match="outside `docs/`"):
        create_rag_system(SystemConfig(type="cli_agent", retrieval={"preset": "codex", "files": {"docs/x.md": "README.md"}}))
    with pytest.raises(ValueError, match="does not exist"):
        create_rag_system(SystemConfig(type="cli_agent", retrieval={"preset": "codex", "files": {"x.md": str(tmp_path / "missing")}}))


def test_workspace_paths_keep_the_corpus_layout_and_never_escape():
    docs = [Document(doc_id=i, path=p, title=i, text="") for i, p in [("a", "/data/c/x/a.md"), ("b", "/data/c/y/b.pdf"), ("c", "/data/c/x/a.md")]]
    assert _workspace_paths(docs) == ["x/a.md", "y/b.pdf.txt", "c/a.md"]
    assert _workspace_paths([Document(doc_id="d", path="../up.md", title="d", text="")]) == ["up.md"]


# -- trajectory metrics --------------------------------------------------------------------------------------------------------------------


def _step(kind, name, query=None, doc_ids=()):
    return {"kind": kind, "name": name, "input_preview": query, "metadata": {"doc_ids": list(doc_ids)}}


def test_redundant_searches_and_steps_to_first_relevant_evidence():
    steps = [_step("tool", "search", "HarborShield price"), _step("tool", "search", "harborshield  PRICE", ["doc_005"]), _step("tool", "read_document", "doc_001", ["doc_001"])]
    assert redundant(steps) == 1 and redundant([_step("retrieve", "s", None), _step("retrieve", "s", None)]) == 0
    assert steps_to_evidence(steps, {"doc_001"}) == 3 and steps_to_evidence(steps, {"doc_009"}) is None
    rows = [
        {"steps": [*steps, _step("generate", "answer")], "relevant_doc_ids": ["doc_001"]},
        {"steps": [_step("tool", "search", "q", ["doc_002"])], "relevant_doc_ids": ["doc_002"]},
        {"steps": [_step("tool", "search", "q")], "relevant_doc_ids": []},  # unanswerable: no evidence to reach
    ]
    assert summary_fields(rows, agentic=True) == {"redundant_search_rate": 1 / 5, "avg_steps_to_evidence": 2.0}
    assert summary_fields(rows, agentic=False) == {}


def _run(tmp_path, dataset, systems) -> tuple[pd.DataFrame, list[dict]]:
    config = {"run": {"name": "cli", "output_dir": str(tmp_path / "results")}, "dataset": dataset, "systems": systems,
              "evaluation": {"max_workers": 2, "latency_probe_questions": 0}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run = run_benchmark(path, force_mock=True, use_cache=False, run_dir=tmp_path / "out")
    return pd.read_csv(run / "metrics_summary.csv").set_index("system"), read_jsonl(run / "per_question_results.jsonl")


def test_mock_e2e_compares_bm25_agent_search_and_a_claude_code_harness(tmp_path, tiny_dataset, monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    systems = [
        {"type": "bm25", "name": "bm25"},
        {"type": "agent_search", "name": "agent_search"},
        {"type": "cli_agent", "name": "claude_code", "retrieval": {"preset": "claude_code", "model": "anthropic:claude-haiku-5-5"}},
    ]
    metrics, rows = _run(tmp_path, tiny_dataset, systems)

    assert metrics.loc["claude_code", "n_error"] == 0 and metrics.loc["claude_code", "retrieval_recall@5"] > 0
    assert metrics.loc["claude_code", "avg_tool_calls"] >= 2 and metrics.loc["claude_code", "avg_llm_calls"] >= 2
    assert metrics.loc["claude_code", "avg_steps_to_evidence"] >= 1 and not pd.isna(metrics.loc["claude_code", "redundant_search_rate"])
    assert not pd.isna(metrics.loc["agent_search", "avg_steps_to_evidence"])  # every agentic system gets the trajectory metrics
    assert pd.isna(metrics.loc["bm25", "redundant_search_rate"]) and pd.isna(metrics.loc["bm25", "avg_steps_to_evidence"])
    harness = [r for r in rows if r["system"] == "claude_code"]
    assert all({"tool", "generate"} <= {step["kind"] for step in r["steps"]} for r in harness)
    assert all(r["agent"]["harness"] == "claude_code" and r["external"]["cost_known"] for r in harness)


def test_a_run_of_harnesses_alone_is_live_and_starts_them_unless_mock_is_asked_for(tmp_path, tiny_dataset):
    """No RAGBench model is called (judge off), yet the harness is: without `--mock` the run must not quietly fall back to the offline one."""
    command = [sys.executable, FAKE, str(FIXTURES / "claude_code.jsonl")]
    system = {"type": "cli_agent", "name": "recorded", "retrieval": {"preset": "custom", "command": command, "output_format": "claude_code", "network": "on"}}
    config = {"run": {"name": "live", "output_dir": str(tmp_path / "results")}, "dataset": tiny_dataset, "systems": [system],
              "evaluation": {"max_workers": 2, "latency_probe_questions": 0, "judge_enabled": False, "max_questions": 2}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    live = read_jsonl(run_benchmark(path, use_cache=False, run_dir=tmp_path / "live") / "per_question_results.jsonl")
    assert [row["cost"]["total_cost"] for row in live] == pytest.approx([0.0012437, 0.0012437])  # the recorded harness ran
    mock = read_jsonl(run_benchmark(path, force_mock=True, use_cache=False, run_dir=tmp_path / "mock") / "per_question_results.jsonl")
    assert all(row["cost"]["total_cost"] < 0.0001 for row in mock)  # the offline harness ran instead


# -- recorded real runs (2026-10-26: Claude Code 2.1.287 on claude-haiku-5-5, Codex 0.162.1 on gpt-6-luna, OpenCode 1.19.4 on z-ai/glm-5.3) ---


@pytest.mark.parametrize(
    ("fmt", "model", "turns", "calls", "read"),
    [
        ("claude_code", "anthropic:claude-haiku-5-5", 4, 6, ["doc_005", "doc_014"]),
        ("codex", "gpt-6-luna", 1, 4, ["doc_017", "doc_014", "doc_005", "doc_057"]),
        ("opencode", "openrouter:z-ai/glm-5.3", 7, 12, ["doc_014", "doc_005", "doc_057"]),
    ],
)
def test_recorded_real_runs_parse_into_steps_passages_and_a_known_cost(fmt, model, turns, calls, read):
    demo = [Document(doc_id=f"doc_{i:03d}", path=f"data/demo/docs/doc_{i:03d}.md", title="", text=f"document {i}") for i in range(1, 61)]
    command = [sys.executable, FAKE, str(FIXTURES / f"{fmt}_recorded.jsonl")]
    system = create_rag_system(SystemConfig(type="cli_agent", name=fmt, retrieval={"preset": "custom", "command": command, "output_format": fmt, "network": "on", "model": model}))
    system.ingest(demo)
    result = system.answer_question("Who is the security lead and what award did they win in 2023?")

    agent = result.metadata["agent"]
    assert (agent["llm_calls"], agent["tool_calls"], agent["outside_workspace_calls"]) == (turns, calls, 0)
    assert result.answer and result.steps[-1].kind == "generate" and result.metadata["external"]["cost_known"] is True
    assert 0 < result.cost.total_cost < 0.02 and sum(step.cost.total_cost for step in result.steps) == pytest.approx(result.cost.total_cost)
    passages = [chunk.doc_id for chunk in result.retrieval_result.chunks]
    assert [doc for doc in passages if doc in read] == read  # every file read is a passage, in the order it was read
    assert not any(step.metadata.get("doc_ids") for step in result.steps if step.metadata.get("action") == "list")
