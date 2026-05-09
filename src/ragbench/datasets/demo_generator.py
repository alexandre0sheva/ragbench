from __future__ import annotations

from pathlib import Path
from typing import Any

from ragbench.utils.jsonl import write_jsonl

DEMO_DOCS: list[tuple[str, str]] = [
    (
        "doc_001.md",
        """# RAGBench Mutual Overview

RAGBench Mutual is a fictional insurance technology company used for benchmark examples. The company builds decision-support tools for carriers, brokers, and claims teams.

Its core products are HarborShield AI for marine risk intake, ClaimPilot for claims triage, and Aurora Risk Suite for portfolio analytics. The customer operations team uses a shared internal knowledge base to track releases, incidents, pricing, and compliance notes.
""",
    ),
    (
        "doc_002.md",
        """# HarborShield AI Product Brief

HarborShield AI reviews marine cargo submissions and extracts vessel route, commodity, port, and sanctions information from broker packets. The product was launched for general availability on 2023-09-18.

HarborShield AI is best suited for marine underwriters who need a first-pass risk memo. It integrates with the Broker Portal and can export a risk narrative into Aurora Risk Suite.
""",
    ),
    (
        "doc_003.md",
        """# ClaimPilot Product Brief

ClaimPilot is a claims triage assistant that summarizes loss notices, classifies severity, and recommends the next handling queue. It is used by property and casualty claims teams.

The production model for ClaimPilot was upgraded in the Solstice 2.1 release. ClaimPilot does not make payment decisions; it only recommends routing and evidence checklists.
""",
    ),
    (
        "doc_004.md",
        """# Aurora Risk Suite

Aurora Risk Suite provides portfolio analytics for underwriting leaders. It compares quote conversion, expected loss, and renewal exposure across regions.

Aurora includes a Scenario Board that lets analysts compare pricing assumptions. Aurora Risk Suite received a dashboard refresh during the Solstice 2.1 release.
""",
    ),
    (
        "doc_005.md",
        """# 2023 Industry Awards

Maxine Thompson won the prestigious Insurance Innovator of the Year award in 2023. The internal shorthand for the award is IIOTY.

The award committee cited her work on HarborShield AI and the internal evaluation program that measured answer faithfulness before customer launch. RAGBench Mutual announced the award on 2023-12-07.
""",
    ),
    (
        "doc_006.md",
        """# Northstar Mutual Case Study

Northstar Mutual adopted ClaimPilot for auto and homeowners claims in April 2024. During the first eight weeks, Northstar reduced average first-touch review time from 18 minutes to 11 minutes.

The case study states that ClaimPilot improved triage consistency but did not change final claim authority. Northstar requested a Spanish-language adjuster note template for a later phase.
""",
    ),
    (
        "doc_007.md",
        """# Meridian Underwriters Case Study

Meridian Underwriters piloted HarborShield AI with its ocean cargo team in November 2023. The pilot focused on broker submissions containing incomplete route details.

Meridian reported a 22 percent reduction in manual intake rework and asked for better flag-state explanations. The account sponsor was Priya Desai.
""",
    ),
    (
        "doc_008.md",
        """# Zurich Pilot Notes

The Zurich pilot evaluated Aurora Risk Suite for European specialty lines. It began on 2024-01-15 and ended on 2024-03-29.

Zurich did not pilot ClaimPilot. The pilot team requested currency normalization and better side-by-side comparison of renewal scenarios.
""",
    ),
    (
        "doc_009.md",
        """# Solstice 2.1 Release Notes

Solstice 2.1 shipped on 2024-02-20. The release upgraded ClaimPilot severity classification, refreshed Aurora dashboards, and added HarborShield export support for the Broker Portal.

Known limitation: HarborShield export does not include sanctions rationale in CSV format. The workaround is to export the PDF risk memo.
""",
    ),
    (
        "doc_010.md",
        """# Compliance Bulletin 2023-11

Compliance Bulletin 2023-11 requires every generated underwriting or claims recommendation to preserve source citations for at least 18 months. The bulletin applies to HarborShield AI, ClaimPilot, and Aurora Risk Suite.

The bulletin also states that AI outputs are advisory and must not be described as final coverage, pricing, or claim payment decisions.
""",
    ),
    (
        "doc_011.md",
        """# Incident Report: API Latency on 2024-02-14

On 2024-02-14, the Broker Portal experienced elevated latency between 09:10 and 10:05 UTC. HarborShield AI requests queued behind a slow document extraction worker.

No customer data was lost. The incident was resolved by scaling the extraction worker pool from 4 to 10 workers. The incident owner was Elias Ren.
""",
    ),
    (
        "doc_012.md",
        """# Pricing Notes for 2024

ClaimPilot is priced per claim notice processed, with a volume discount above 250,000 notices per year. HarborShield AI is priced per marine submission packet.

Aurora Risk Suite uses a seat-based subscription. The finance team retired the old Aurora Lite SKU on 2024-01-31 and replaced it with Aurora Team.
""",
    ),
    (
        "doc_013.md",
        """# Support Escalation Policy

Severity 1 support issues require customer acknowledgement within 30 minutes and an incident commander within 45 minutes. Severity 2 issues require acknowledgement within 4 business hours.

For AI answer quality escalations, support must attach the question, retrieved context, model answer, customer impact, and the product name.
""",
    ),
    (
        "doc_014.md",
        """# Employee Profile: Maxine Thompson

Maxine Thompson is Vice President of Applied AI at RAGBench Mutual. She sponsored the internal faithfulness evaluation program and led the launch review for HarborShield AI.

Maxine previously worked in marine underwriting analytics. She won the Insurance Innovator of the Year award in 2023.
""",
    ),
    (
        "doc_015.md",
        """# Employee Profile: Elias Ren

Elias Ren is the reliability engineering lead for the Broker Portal and document extraction services. He owned the 2024-02-14 API latency incident.

Elias also maintains the runbook for extraction worker autoscaling and queue-depth alerts.
""",
    ),
    (
        "doc_016.md",
        """# Data Retention Policy

RAGBench Mutual retains source citations for AI recommendations for 18 months. Raw uploaded broker packets are retained for 90 days unless a customer contract requires shorter retention.

Evaluation traces used in offline benchmarks may be retained for 12 months when they contain no production customer data.
""",
    ),
    (
        "doc_017.md",
        """# Security Architecture Summary

The Broker Portal stores documents in encrypted object storage. HarborShield AI, ClaimPilot, and Aurora Risk Suite access documents through scoped service tokens.

All production AI requests include tenant isolation metadata. Security review requires audit logs for retrieval, prompt construction, model response, and user export.
""",
    ),
    (
        "doc_018.md",
        """# Broker Portal FAQ

The Broker Portal accepts PDF, DOCX, TXT, and email-body submissions. HarborShield AI export is available as PDF in Solstice 2.1, while CSV export omits sanctions rationale.

If the portal is slow, users should check the status page before retrying large uploads.
""",
    ),
    (
        "doc_019.md",
        """# 2024 Product Roadmap

The 2024 roadmap includes multilingual adjuster notes for ClaimPilot, renewal scenario comparison for Aurora Risk Suite, and expanded sanctions rationale export for HarborShield AI.

The roadmap labels the Spanish-language adjuster note template as a second-half 2024 item.
""",
    ),
    (
        "doc_020.md",
        """# Product Naming Memo

The commercial team proposed renaming Aurora Lite to HarborShield Essentials in an early January 2024 draft. That draft was not approved.

The final pricing decision retired Aurora Lite and replaced it with Aurora Team, as recorded by finance on 2024-01-31.
""",
    ),
    (
        "doc_021.md",
        """# Claims Triage Playbook

ClaimPilot severity labels are low, medium, high, and critical. Critical claims include possible fatality, major litigation exposure, or regulatory reporting deadlines.

Adjusters must verify ClaimPilot summaries against the original loss notice before contacting the insured.
""",
    ),
    (
        "doc_022.md",
        """# Partner Integrations

HarborShield AI integrates with the Broker Portal and exports risk memos to Aurora Risk Suite. ClaimPilot integrates with Guidewire ClaimCenter through a queue-routing adapter.

Aurora Risk Suite imports pricing assumptions from CSV and exports portfolio summaries as XLSX.
""",
    ),
]


def q(
    qid: str,
    question: str,
    answer: str,
    keywords: list[str],
    docs: list[str],
    category: str,
    difficulty: str = "easy",
    answer_type: str = "single_fact",
) -> dict[str, Any]:
    return {
        "id": qid,
        "question": question,
        "reference_answer": answer,
        "expected_keywords": keywords,
        "relevant_doc_ids": docs,
        "category": category,
        "difficulty": difficulty,
        "answer_type": answer_type,
    }


DEMO_QUESTIONS: list[dict[str, Any]] = [
    q("q_001", "Who won the prestigious IIOTY award in 2023?", "Maxine Thompson won the Insurance Innovator of the Year award in 2023.", ["Maxine Thompson", "Insurance Innovator of the Year", "2023"], ["doc_005", "doc_014"], "direct_fact"),
    q("q_002", "What product reviews marine cargo submissions?", "HarborShield AI reviews marine cargo submissions.", ["HarborShield AI", "marine cargo submissions"], ["doc_002"], "direct_fact"),
    q("q_003", "When did HarborShield AI become generally available?", "HarborShield AI launched for general availability on 2023-09-18.", ["2023-09-18", "general availability"], ["doc_002"], "date_or_timeline"),
    q("q_004", "Which product is priced per claim notice processed?", "ClaimPilot is priced per claim notice processed.", ["ClaimPilot", "per claim notice"], ["doc_012"], "direct_fact"),
    q("q_005", "What are ClaimPilot severity labels?", "ClaimPilot severity labels are low, medium, high, and critical.", ["low", "medium", "high", "critical"], ["doc_021"], "list_answer", answer_type="list"),
    q("q_006", "Which customer requested a Spanish-language adjuster note template?", "Northstar Mutual requested a Spanish-language adjuster note template.", ["Northstar Mutual", "Spanish-language adjuster note template"], ["doc_006", "doc_019"], "direct_fact"),
    q("q_007", "How much did Northstar Mutual reduce first-touch review time?", "Northstar reduced average first-touch review time from 18 minutes to 11 minutes.", ["18 minutes", "11 minutes", "Northstar"], ["doc_006"], "numeric_answer"),
    q("q_008", "Which company piloted HarborShield AI in November 2023?", "Meridian Underwriters piloted HarborShield AI in November 2023.", ["Meridian Underwriters", "November 2023"], ["doc_007"], "direct_fact"),
    q("q_009", "What reduction did Meridian report during the HarborShield pilot?", "Meridian reported a 22 percent reduction in manual intake rework.", ["22 percent", "manual intake rework"], ["doc_007"], "numeric_answer"),
    q("q_010", "Did Zurich pilot ClaimPilot?", "No. Zurich evaluated Aurora Risk Suite and did not pilot ClaimPilot.", ["Zurich", "did not pilot ClaimPilot", "Aurora Risk Suite"], ["doc_008"], "entity_disambiguation"),
    q("q_011", "When did the Zurich pilot begin and end?", "The Zurich pilot began on 2024-01-15 and ended on 2024-03-29.", ["2024-01-15", "2024-03-29"], ["doc_008"], "date_or_timeline"),
    q("q_012", "What shipped in Solstice 2.1?", "Solstice 2.1 upgraded ClaimPilot severity classification, refreshed Aurora dashboards, and added HarborShield export support for the Broker Portal.", ["ClaimPilot", "Aurora dashboards", "HarborShield export", "Broker Portal"], ["doc_009"], "list_answer", answer_type="list"),
    q("q_013", "What HarborShield export limitation exists in Solstice 2.1?", "HarborShield CSV export does not include sanctions rationale; the PDF risk memo is the workaround.", ["CSV", "sanctions rationale", "PDF risk memo"], ["doc_009", "doc_018"], "direct_fact"),
    q("q_014", "Which compliance bulletin requires source citations for 18 months?", "Compliance Bulletin 2023-11 requires source citations for at least 18 months.", ["Compliance Bulletin 2023-11", "18 months"], ["doc_010", "doc_016"], "direct_fact"),
    q("q_015", "What products are covered by Compliance Bulletin 2023-11?", "The bulletin applies to HarborShield AI, ClaimPilot, and Aurora Risk Suite.", ["HarborShield AI", "ClaimPilot", "Aurora Risk Suite"], ["doc_010"], "list_answer", answer_type="list"),
    q("q_016", "Who owned the 2024-02-14 API latency incident?", "Elias Ren owned the 2024-02-14 API latency incident.", ["Elias Ren", "2024-02-14"], ["doc_011", "doc_015"], "direct_fact"),
    q("q_017", "How was the 2024-02-14 API latency incident resolved?", "It was resolved by scaling the extraction worker pool from 4 to 10 workers.", ["4", "10", "extraction worker"], ["doc_011"], "numeric_answer"),
    q("q_018", "How long must raw uploaded broker packets be retained?", "Raw uploaded broker packets are retained for 90 days unless a customer contract requires shorter retention.", ["90 days", "broker packets"], ["doc_016"], "direct_fact"),
    q("q_019", "What is the acknowledgement target for Severity 1 support issues?", "Severity 1 issues require customer acknowledgement within 30 minutes.", ["Severity 1", "30 minutes"], ["doc_013"], "direct_fact"),
    q("q_020", "What evidence should support attach for AI answer quality escalations?", "Support must attach the question, retrieved context, model answer, customer impact, and product name.", ["question", "retrieved context", "model answer", "customer impact", "product name"], ["doc_013"], "list_answer", answer_type="list"),
    q("q_021", "What is Maxine Thompson's role?", "Maxine Thompson is Vice President of Applied AI at RAGBench Mutual.", ["Vice President of Applied AI", "Maxine Thompson"], ["doc_014"], "direct_fact"),
    q("q_022", "What runbook does Elias Ren maintain?", "Elias Ren maintains the runbook for extraction worker autoscaling and queue-depth alerts.", ["extraction worker autoscaling", "queue-depth alerts"], ["doc_015"], "direct_fact"),
    q("q_023", "What file types does the Broker Portal accept?", "The Broker Portal accepts PDF, DOCX, TXT, and email-body submissions.", ["PDF", "DOCX", "TXT", "email-body"], ["doc_018"], "list_answer", answer_type="list"),
    q("q_024", "Which product integrates with Guidewire ClaimCenter?", "ClaimPilot integrates with Guidewire ClaimCenter through a queue-routing adapter.", ["ClaimPilot", "Guidewire ClaimCenter"], ["doc_022"], "direct_fact"),
    q("q_025", "Which product exports risk memos to Aurora Risk Suite?", "HarborShield AI exports risk memos to Aurora Risk Suite.", ["HarborShield AI", "risk memos", "Aurora Risk Suite"], ["doc_022", "doc_002"], "direct_fact"),
    q("q_026", "Compare ClaimPilot and HarborShield AI by primary workflow.", "ClaimPilot supports claims triage, while HarborShield AI supports marine underwriting intake.", ["ClaimPilot", "claims triage", "HarborShield AI", "marine"], ["doc_002", "doc_003"], "comparison", difficulty="medium"),
    q("q_027", "Which two documents together show Maxine's award and her role?", "The awards document and Maxine Thompson profile show she won IIOTY and is VP of Applied AI.", ["doc_005", "doc_014", "VP of Applied AI"], ["doc_005", "doc_014"], "multi_hop", difficulty="medium"),
    q("q_028", "What changed for ClaimPilot in Solstice 2.1 and what labels does ClaimPilot use?", "Solstice 2.1 upgraded ClaimPilot severity classification, and the labels are low, medium, high, and critical.", ["Solstice 2.1", "severity classification", "low", "critical"], ["doc_009", "doc_021"], "multi_hop", difficulty="medium", answer_type="list"),
    q("q_029", "Which customer wanted renewal scenario comparison improvements?", "Zurich requested better side-by-side comparison of renewal scenarios.", ["Zurich", "renewal scenarios"], ["doc_008", "doc_019"], "direct_fact"),
    q("q_030", "What second-half 2024 roadmap item relates to Northstar's request?", "The Spanish-language adjuster note template is labeled as a second-half 2024 item.", ["Spanish-language adjuster note template", "second-half 2024"], ["doc_006", "doc_019"], "multi_hop", difficulty="medium"),
    q("q_031", "What was the final replacement for Aurora Lite?", "The final decision replaced Aurora Lite with Aurora Team.", ["Aurora Lite", "Aurora Team"], ["doc_012", "doc_020"], "conflicting_information"),
    q("q_032", "Was HarborShield Essentials the approved replacement for Aurora Lite?", "No. HarborShield Essentials was only an early draft proposal; the approved replacement was Aurora Team.", ["HarborShield Essentials", "draft", "Aurora Team"], ["doc_020", "doc_012"], "conflicting_information", difficulty="medium"),
    q("q_033", "What audit logs are required by security review?", "Security review requires audit logs for retrieval, prompt construction, model response, and user export.", ["retrieval", "prompt construction", "model response", "user export"], ["doc_017"], "list_answer", answer_type="list"),
    q("q_034", "How are production AI requests isolated?", "Production AI requests include tenant isolation metadata.", ["tenant isolation metadata"], ["doc_017"], "direct_fact"),
    q("q_035", "What should users check if the Broker Portal is slow?", "Users should check the status page before retrying large uploads.", ["status page", "retrying large uploads"], ["doc_018"], "direct_fact"),
    q("q_036", "Which products can export or import files according to partner integrations?", "Aurora imports CSV and exports XLSX, while HarborShield exports risk memos to Aurora.", ["CSV", "XLSX", "HarborShield", "risk memos"], ["doc_022"], "summary", difficulty="medium"),
    q("q_037", "Summarize the advisory limitation across AI products.", "AI outputs are advisory and must not be described as final coverage, pricing, or claim payment decisions.", ["advisory", "final coverage", "pricing", "claim payment"], ["doc_010", "doc_003"], "summary", difficulty="medium"),
    q("q_038", "Which product should a marine underwriter use for a first-pass risk memo?", "A marine underwriter should use HarborShield AI for a first-pass risk memo.", ["marine underwriter", "HarborShield AI", "risk memo"], ["doc_002"], "direct_fact"),
    q("q_039", "What product has a Scenario Board?", "Aurora Risk Suite has a Scenario Board.", ["Aurora Risk Suite", "Scenario Board"], ["doc_004"], "direct_fact"),
    q("q_040", "What happened to customer data during the 2024-02-14 incident?", "No customer data was lost during the incident.", ["No customer data was lost"], ["doc_011"], "direct_fact"),
    q("q_041", "Who is the CEO of RAGBench Mutual?", "The provided documents do not state who the CEO of RAGBench Mutual is.", [], [], "not_in_context", answer_type="unanswerable"),
    q("q_042", "What is the exact annual revenue of RAGBench Mutual?", "The provided documents do not include exact annual revenue.", [], [], "not_in_context", answer_type="unanswerable"),
    q("q_043", "Which cloud provider hosts HarborShield AI?", "The provided documents do not identify the cloud provider hosting HarborShield AI.", [], [], "not_in_context", answer_type="unanswerable"),
    q("q_044", "What was ClaimPilot's accuracy score in 2022?", "The provided documents do not report a 2022 ClaimPilot accuracy score.", [], [], "not_in_context", answer_type="unanswerable"),
    q("q_045", "Which office is Priya Desai based in?", "The provided documents name Priya Desai as the Meridian account sponsor but do not state her office location.", ["Priya Desai"], [], "not_in_context", answer_type="unanswerable"),
]


def write_demo_dataset(base_dir: Path = Path("data/demo"), overwrite: bool = False) -> dict[str, int]:
    docs_dir = base_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    written_docs = 0
    for filename, content in DEMO_DOCS:
        path = docs_dir / filename
        if overwrite or not path.exists():
            path.write_text(content.strip() + "\n", encoding="utf-8")
            written_docs += 1
    write_jsonl(base_dir / "questions.jsonl", DEMO_QUESTIONS)
    qrels = []
    for question in DEMO_QUESTIONS:
        for idx, doc_id in enumerate(question["relevant_doc_ids"]):
            qrels.append({"query_id": question["id"], "doc_id": doc_id, "relevance": 3 if idx == 0 else 2})
    write_jsonl(base_dir / "qrels.jsonl", qrels)
    return {"documents": len(DEMO_DOCS), "questions": len(DEMO_QUESTIONS), "qrels": len(qrels), "written_docs": written_docs}
