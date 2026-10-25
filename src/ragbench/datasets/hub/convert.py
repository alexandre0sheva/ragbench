"""Converters from a hub dataset's source rows to RAGBench's documents, questions and qrels.

Each takes the rows exactly as the source publishes them and returns a `Converted`; nothing here touches the network or the disk.
Documents are named `doc_*` so their ids survive as file stems (see `stable_doc_id`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ragbench.utils.hashing import stable_hash


@dataclass(frozen=True)
class ConvertedDoc:
    title: str
    text: str


@dataclass
class Converted:
    documents: dict[str, ConvertedDoc] = field(default_factory=dict)  # doc id -> document
    questions: list[dict[str, Any]] = field(default_factory=list)  # rows of questions.jsonl
    qrels: list[dict[str, Any]] = field(default_factory=list)  # rows of qrels.jsonl
    # Datasets whose questions each come with their own passages (HotpotQA, RAGBench): question id -> the documents shown with it.
    # Empty when one corpus serves every question. A subset keeps only the documents of the questions it keeps.
    contexts: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _squash(text: str) -> str:
    return " ".join(text.lower().split())


def _slug(text: str, limit: int = 30) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:limit] or "untitled"


def _qrels(question_id: str, doc_ids: list[str]) -> list[dict[str, Any]]:
    return [{"query_id": question_id, "doc_id": doc_id, "relevance": 1} for doc_id in doc_ids]


# -- MultiHop-RAG: a news corpus and questions whose evidence is a list of articles ---------------


def convert_multihop_rag(corpus: list[dict[str, Any]], queries: list[dict[str, Any]]) -> Converted:
    """Articles keep their source and date in the text (the temporal questions need them). An evidence item is matched to its article by
    url, then by title; its `fact` is kept as `evidence` only when it appears in the article. A `null_query` is unanswerable."""
    converted = Converted()
    by_url: dict[str, str] = {}
    by_title: dict[str, str] = {}
    bodies: dict[str, str] = {}
    for number, article in enumerate(corpus, start=1):
        doc_id = f"doc_news_{number:04d}"
        title = str(article.get("title") or "").strip()
        byline = " · ".join(str(article[key]).strip() for key in ("source", "published_at", "author") if article.get(key))
        body = str(article.get("body") or "").strip()
        converted.documents[doc_id] = ConvertedDoc(title, f"{byline}\n\n{body}" if byline else body)
        by_url.setdefault(str(article.get("url") or ""), doc_id)
        by_title.setdefault(_squash(title), doc_id)
        bodies[doc_id] = _squash(body)
    for number, query in enumerate(queries, start=1):
        question_id = f"mh_{number:04d}"
        relevant: list[str] = []
        evidence: list[dict[str, str]] = []
        unmatched = 0
        for item in query.get("evidence_list") or []:
            found = by_url.get(str(item.get("url") or "")) or by_title.get(_squash(str(item.get("title") or "")))
            if found is None:
                unmatched += 1
                continue
            if found not in relevant:
                relevant.append(found)
            fact = str(item.get("fact") or "").strip()
            if fact and _squash(fact) in bodies[found] and {"doc_id": found, "quote": fact} not in evidence:
                evidence.append({"doc_id": found, "quote": fact})
        if unmatched:
            converted.warnings.append(f"{question_id}: {unmatched} evidence article(s) are not in the corpus; left out of its labels")
        null = query.get("question_type") == "null_query"
        row: dict[str, Any] = {"id": question_id, "question": str(query["query"]).strip(), "reference_answer": str(query.get("answer") or "").strip() or None}
        row = {key: value for key, value in row.items() if value is not None}
        if relevant:
            row["relevant_doc_ids"] = relevant
        row["answerable"] = not null
        row["category"] = str(query.get("question_type") or "unknown")
        if evidence:
            row["evidence"] = evidence
        converted.questions.append(row)
        converted.qrels.extend(_qrels(question_id, relevant))
    return converted


# -- HotpotQA (distractor): ten paragraphs per question, two of them the supporting ones --------------


def convert_hotpotqa(rows: list[dict[str, Any]]) -> Converted:
    """One document per distinct paragraph (same title and text = same document). The supporting facts give the relevant documents and,
    sentence by sentence, the evidence quotes."""
    converted = Converted()
    for row in rows:
        context = row["context"]
        titles, sentences = list(context["title"]), list(context["sentences"])
        shown: dict[str, str] = {}  # paragraph title -> doc id, within this question
        for title, parts in zip(titles, sentences, strict=True):
            text = " ".join(part.strip() for part in parts)
            digest = stable_hash(title + "\n" + text, 6)
            doc_id = f"doc_hp_{_slug(title)}_{digest}"
            converted.documents.setdefault(doc_id, ConvertedDoc(title, text))
            shown.setdefault(title, doc_id)
        question_id = str(row["id"])
        facts = row["supporting_facts"]
        relevant: list[str] = []
        evidence: list[dict[str, str]] = []
        for title, sentence_id in zip(facts["title"], facts["sent_id"], strict=True):
            found = shown.get(title)
            if found is None:
                converted.warnings.append(f"{question_id}: supporting paragraph {title!r} is not in its context; left out")
                continue
            if found not in relevant:
                relevant.append(found)
            position = titles.index(title)
            if 0 <= sentence_id < len(sentences[position]):
                evidence.append({"doc_id": found, "quote": sentences[position][sentence_id].strip()})
        converted.questions.append(
            {
                "id": question_id,
                "question": str(row["question"]).strip(),
                "reference_answer": str(row["answer"]).strip(),
                "relevant_doc_ids": relevant,
                "answerable": True,
                "category": str(row.get("type") or "unknown"),
                "difficulty": str(row.get("level") or "unknown"),
                **({"evidence": evidence} if evidence else {}),
            }
        )
        converted.qrels.extend(_qrels(question_id, relevant))
        converted.contexts[question_id] = list(dict.fromkeys(shown.values()))
    return converted


# -- Galileo's RAGBench: a question, the passages a RAG system was given, an annotated response --------

_KEY_INDEX = re.compile(r"^(\d+)")


def convert_galileo(rows: list[dict[str, Any]], subset: str) -> Converted:
    """Passages are shared by text. A passage is relevant when the annotators marked one of its sentences relevant; a question with none
    is unanswerable from its passages. The response becomes the reference answer only when the annotators found it fully supported."""
    converted = Converted()
    seen: set[str] = set()
    for row in rows:
        question_id = str(row["id"])
        if question_id in seen:
            unique = f"{question_id}-{len(seen)}"
            converted.warnings.append(f"Duplicate question id {question_id!r} in the source; the later one is called {unique!r}")
            question_id = unique
        seen.add(question_id)
        doc_ids: list[str] = []
        for text in row["documents"]:
            doc_id = f"doc_{_slug(subset, 12)}_{stable_hash(text, 10)}"
            converted.documents.setdefault(doc_id, ConvertedDoc("", text.strip()))
            doc_ids.append(doc_id)
        relevant: list[str] = []
        for key in row.get("all_relevant_sentence_keys") or []:
            match = _KEY_INDEX.match(str(key))
            if match and int(match.group(1)) < len(doc_ids) and doc_ids[int(match.group(1))] not in relevant:
                relevant.append(doc_ids[int(match.group(1))])
        response = str(row.get("response") or "").strip()
        adherent = bool(row.get("adherence_score")) and bool(response)
        out: dict[str, Any] = {"id": question_id, "question": str(row["question"]).strip()}
        if adherent:
            out["reference_answer"] = response
        if relevant:
            out["relevant_doc_ids"] = relevant
        out["answerable"] = bool(relevant)
        out["category"] = str(row.get("dataset_name") or subset).rsplit("_", 1)[0] or subset
        if adherent:
            out["metadata"] = {"reference_is_model_response": True}
        converted.questions.append(out)
        converted.qrels.extend(_qrels(question_id, relevant))
        converted.contexts[question_id] = list(dict.fromkeys(doc_ids))
    return converted
