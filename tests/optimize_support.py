"""A dataset and system whose quality a harness edit visibly changes, for the optimizer tests.

Questions ask for the code of an item: `alpha` items are the train and dev questions, `omega` items the test questions. The `fake_tunable`
system answers by its `answer_instructions`: by default it knows the codes of even-numbered items only; with `GOOD` it knows every code;
with `MAGIC` it knows every `alpha` code and no `omega` code (an edit that helps on train and dev and hurts on test). Each answer costs
`ANSWER_COST`, so spending caps bite in mock runs too.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml
from fake_systems import RankedFakeSystem, write_experiment

from ragbench.datasets.splits import Splits, write_splits
from ragbench.models.cost import CostBreakdown
from ragbench.models.llms import LLMResult
from ragbench.rag_systems import SYSTEM_REGISTRY

ANSWER_COST = 0.001
TRAIN, DEV, TEST = range(1, 17), range(17, 25), range(1, 17)


class TunableFakeSystem(RankedFakeSystem):
    def _generate_answer(self, question: str, chunks):
        match = re.search(r"item (alpha|omega) (\d+)", question)
        group, number = (match[1], int(match[2])) if match else ("", 1)
        instructions = self.harness.answer_instructions or ""
        knows = True if "GOOD" in instructions else group == "alpha" if "MAGIC" in instructions else number % 2 == 0
        text = f"The code is code{number}x." if knows else "The code is something else."
        with self.trace.step("generate", "answer", context_chunks=len(chunks)) as step:
            result = LLMResult(text, "fake-model", 10, 5, CostBreakdown(llm_cost=ANSWER_COST), finish_reason="stop")
            step.set_llm(result)
            step.set_input(question)
        return result


def register_tunable(monkeypatch) -> None:
    monkeypatch.setitem(SYSTEM_REGISTRY, "fake_tunable", TunableFakeSystem)


def _question(group: str, number: int) -> dict:
    return {
        "id": f"item_{group}_{number:03d}",
        "question": f"What is the code of item {group} {number}?",
        "reference_answer": f"code{number}x",
        "expected_keywords": [f"code{number}x"],
        "relevant_doc_ids": ["doc_001"],
        "category": "direct_fact",
    }


def write_project(root: Path, *, dev: bool = True, evaluation: dict | None = None) -> Path:
    """The config of a one-system project on the item dataset, with `splits.json` next to the questions."""
    questions = [*(_question("alpha", n) for n in TRAIN), *(_question("alpha", n) for n in DEV), *(_question("omega", n) for n in TEST)]
    systems = [{"type": "fake_tunable", "name": "tuned", "retrieval": {"top_k": 3}}]
    path = write_experiment(root, questions, systems, evaluation={"latency_probe_questions": 0, **(evaluation or {})})
    splits = Splits(
        seed=0,
        train=[f"item_alpha_{n:03d}" for n in TRAIN],
        dev=[f"item_alpha_{n:03d}" for n in DEV] if dev else [],
        test=[f"item_omega_{n:03d}" for n in TEST] + ([] if dev else [f"item_alpha_{n:03d}" for n in DEV]),
    )
    write_splits(root / "splits.json", splits)
    return path


def edit_config(path: Path, edit) -> None:
    path.write_text(yaml.safe_dump(edit(yaml.safe_load(path.read_text(encoding="utf-8")))), encoding="utf-8")


def scripted_reply(prompt: str, edits: dict[str, str]) -> str:
    """A proposer reply: `edits[kind]` as `answer_instructions` (`""` for an empty patch)."""
    request = prompt.rsplit("=== request ===", 1)[-1]
    kind = re.search(r"^Kind: (\w+)", request, flags=re.M)[1]  # type: ignore[index]
    value = edits.get(kind, "")
    return json.dumps({"rationale": f"scripted {kind}", "harness": {"answer_instructions": value} if value else {}})
