"""One tiny dataset for tests that need a real corpus: six short documents and eight questions of the kinds the systems treat differently."""

from __future__ import annotations

from pathlib import Path

from ragbench.utils.jsonl import write_jsonl

DOCUMENTS = {
    "doc_001": "# Pricing\n\n## Plans\n\nHarborShield costs $200 per month for the marine module. The enterprise plan costs $950 per month.\n",
    "doc_002": "# Roadmap\n\nClaimPilot ships in Q3 with claims triage workflows. The beta started on 2024-01-15 and general availability is 2024-03-29.\n",
    "doc_003": "# Support\n\nSupport is available around the clock for every plan. Error HS-4127 means the tenant header is missing.\n",
    "doc_004": "# Security\n\nAll data is encrypted at rest. Audit logs are kept for 400 days. The security lead is Maxine Thompson.\n",
    "doc_005": "# Awards\n\nMaxine Thompson won the Insurance Innovator of the Year award in 2023 for the HarborShield launch.\n",
    "doc_006": "# Offices\n\nThe Zurich office opened in 2022. The Lisbon office opened in 2024 and hosts the claims team.\n",
}

QUESTIONS = [
    {"id": "q1", "question": "How much does HarborShield cost per month?", "reference_answer": "$200 per month.", "expected_keywords": ["$200"], "relevant_doc_ids": ["doc_001"], "category": "direct_fact"},
    {"id": "q2", "question": "When does ClaimPilot ship?", "reference_answer": "Q3.", "expected_keywords": ["Q3"], "relevant_doc_ids": ["doc_002"], "category": "direct_fact"},
    {"id": "q3", "question": "What does error HS-4127 mean?", "reference_answer": "The tenant header is missing.", "expected_keywords": ["tenant header"], "relevant_doc_ids": ["doc_003"], "category": "exact_match"},
    {"id": "q4", "question": "Who is the security lead and what award did they win in 2023?", "reference_answer": "Maxine Thompson won the Insurance Innovator of the Year award.", "expected_keywords": ["Maxine Thompson"], "relevant_doc_ids": ["doc_004", "doc_005"], "category": "multi_hop"},
    {"id": "q5", "question": "How many days are there between the ClaimPilot beta start and general availability?", "reference_answer": "74 days.", "expected_keywords": ["74"], "relevant_doc_ids": ["doc_002"], "category": "date_arithmetic", "requires_tools": ["date_calc"]},
    {"id": "q6", "question": "What is the enterprise plan price multiplied by 12?", "reference_answer": "$11,400.", "expected_keywords": ["11400"], "relevant_doc_ids": ["doc_001"], "category": "numeric", "requires_tools": ["calculator"]},
    {"id": "q7", "question": "Which office hosts the claims team?", "reference_answer": "The Lisbon office.", "expected_keywords": ["Lisbon"], "relevant_doc_ids": ["doc_006"], "category": "direct_fact"},
    {"id": "q8", "question": "Who is the CEO of Atlantis Corp?", "reference_answer": "The documents do not say.", "answerable": False, "category": "unanswerable"},
]


def write_tiny_dataset(root: Path, *, category: str | None = None) -> dict[str, str]:
    """Write the corpus and questions under `root`; returns the `dataset:` section of a config pointing at them.

    `category` replaces the category of every answerable question (for tests that need a particular, or hostile, category name)."""
    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    for doc_id, text in DOCUMENTS.items():
        (docs / f"{doc_id}.md").write_text(text, encoding="utf-8")
    write_jsonl(root / "questions.jsonl", [q if category is None or q.get("answerable") is False else {**q, "category": category} for q in QUESTIONS])
    return {"documents_path": str(docs), "questions_path": str(root / "questions.jsonl")}


def write_pair_dataset(root: Path, ids: tuple[str, str] = ("q1", "q2")) -> dict[str, str]:
    """The smallest labeled corpus: two documents and one question about each (for tests of run mechanics, not of retrieval)."""
    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "doc_001.md").write_text(DOCUMENTS["doc_001"].split("\n\n## Plans")[0] + "\n\nHarborShield costs $200 per month for the marine module.\n", encoding="utf-8")
    (docs / "doc_002.md").write_text("# Roadmap\n\nClaimPilot ships in Q3 with claims triage workflows.\n", encoding="utf-8")
    write_jsonl(
        root / "questions.jsonl",
        [
            {"id": ids[0], "question": "How much does HarborShield cost?", "reference_answer": "$200 per month.", "relevant_doc_ids": ["doc_001"], "category": "direct_fact"},
            {"id": ids[1], "question": "When does ClaimPilot ship?", "reference_answer": "Q3.", "relevant_doc_ids": ["doc_002"], "category": "direct_fact"},
        ],
    )
    return {"documents_path": str(docs), "questions_path": str(root / "questions.jsonl")}
