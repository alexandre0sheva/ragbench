"""Every prompt the agentic systems send, in one place. Each carries a marker phrase (`models/prompts.py`) the mock model recognises."""

from __future__ import annotations

from typing import Any

from ragbench.models.prompts import (
    CHECK_GROUNDED_MARKER,
    FORCE_ANSWER_MARKER,
    GRADE_CHUNKS_MARKER,
    NEXT_HOP_MARKER,
    REACT_JSON_MARKER,
    REWRITE_QUERY_MARKER,
    ROUTE_QUESTION_MARKER,
    TOOL_AGENT_MARKER,
)

Messages = list[dict[str, Any]]
PASSAGE_CHARS = 700


def _passage(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= PASSAGE_CHARS else text[: PASSAGE_CHARS - 1] + "…"


def grade_chunks(question: str, passages: list[str]) -> Messages:
    numbered = "\n".join(f"[{number}] {_passage(text)}" for number, text in enumerate(passages, start=1))
    return [
        {"role": "system", "content": "You judge whether retrieved passages help answer a question. Answer with JSON only."},
        {
            "role": "user",
            "content": (
                f"{GRADE_CHUNKS_MARKER}: relevant (states information that answers the question or a needed part of it), "
                "ambiguous (related but incomplete or unclear) or irrelevant (unrelated).\n"
                f"Question: {question}\n\nPassages:\n{numbered}\n\n"
                'Return JSON like {"grades": [{"id": 1, "grade": "relevant"}, {"id": 2, "grade": "irrelevant"}]} with one entry per passage.'
            ),
        },
    ]


def rewrite_query(question: str, previous_query: str, found: str) -> Messages:
    return [
        {"role": "system", "content": "You write search queries for a document search system. Answer with JSON only."},
        {
            "role": "user",
            "content": (
                f"{REWRITE_QUERY_MARKER}. The previous query did not retrieve passages that answer the question. "
                "Use different wording or synonyms, and name the specific entities and facts that are needed.\n"
                f"Question: {question}\nPrevious query: {previous_query}\nWhat the search returned instead: {found}\n"
                'Return JSON like {"query": "..."}.'
            ),
        },
    ]


def next_hop(question: str, evidence: str, hop: int, max_hops: int) -> Messages:
    return [
        {"role": "system", "content": "You research a question step by step in a document collection. Answer with JSON only."},
        {
            "role": "user",
            "content": (
                f"{NEXT_HOP_MARKER}. Say what the evidence shows so far and what is still missing. If the evidence is enough to answer the "
                "question, set next_query to DONE; otherwise write the next search query for the missing information (a new query, not a repeat). "
                f"This was search round {hop} of at most {max_hops}.\n"
                f"Question: {question}\n\nEvidence so far:\n{evidence}\n\n"
                'Return JSON like {"known": "...", "missing": "...", "next_query": "... or DONE"}.'
            ),
        },
    ]


def check_grounded(question: str, context: str, answer: str) -> Messages:
    return [
        {"role": "system", "content": "You check answers against their sources. Answer with JSON only."},
        {
            "role": "user",
            "content": (
                f"{CHECK_GROUNDED_MARKER}. List every claim in the answer that the context does not support.\n"
                f"Question: {question}\n\nContext:\n{context}\n\nAnswer: {answer}\n\n"
                'Return JSON like {"supported": true, "unsupported_claims": []}.'
            ),
        },
    ]


def regenerate_note(unsupported_claims: list[str]) -> str:
    """Appended to the answer prompt's system message when a draft was found to contain unsupported claims."""
    listed = "; ".join(unsupported_claims) if unsupported_claims else "some of its claims"
    return f"\nA previous draft of this answer contained claims the context does not support ({listed}). Answer again using only what the context states."


def tool_agent_system(react_tools: str | None = None, instructions: str = "") -> str:
    """System prompt of the tool-using agents. `react_tools` (a description of the tools) switches on the JSON-action protocol for models without native tool calling."""
    text = (
        f"{TOOL_AGENT_MARKER}, answering a question about a collection of documents.\n"
        "Use the tools to find the facts you need, then answer. Rules:\n"
        "- Base the answer only on what the tools returned; never guess or use outside knowledge.\n"
        "- If a tool can compute something exactly (arithmetic, dates), use it instead of working it out yourself.\n"
        "- Cite the source document ID in square brackets, for example [doc_014], after each fact taken from a document.\n"
        '- If the documents do not contain the answer, say: "I could not find the answer in the provided documents."\n'
        "- When you have enough, reply with the final answer as plain text and no tool call. Be concise but complete."
    )
    if instructions:
        text += "\n" + instructions
    if react_tools is not None:
        text += (
            f"\n\nYou cannot call functions directly. {REACT_JSON_MARKER} and nothing else each turn.\n"
            'To use a tool: {"action": "tool", "tool": "<tool name>", "args": {<arguments>}}\n'
            'To give the final answer: {"action": "answer", "answer": "<your answer with [doc_id] citations>"}\n'
            "After each tool call you will receive its result as an observation.\n\nAvailable tools:\n" + react_tools
        )
    return text


def question_message(question: str) -> str:
    return f"Question: {question}"


def force_answer() -> str:
    return (
        f"{FORCE_ANSWER_MARKER}: the step or budget limit was reached. Answer the question now using only what the tools returned, "
        "citing document IDs in square brackets. If it is not enough to answer, say so."
    )


def observation(tool: str, text: str) -> str:
    """A tool result as a user message, for ReAct-JSON mode."""
    return f"Observation from {tool}:\n{text}"


def route_question(question: str, routes: dict[str, str]) -> Messages:
    """`routes` maps each route name to a one-line description of the questions it is for."""
    listed = "\n".join(f"- {name}: {description}" for name, description in routes.items())
    return [
        {"role": "system", "content": "You route questions to the retrieval pipeline best suited to them. Answer with JSON only."},
        {
            "role": "user",
            "content": (
                f"{ROUTE_QUESTION_MARKER}. Pick exactly one of the routes and say why in a few words.\n"
                f"Routes:\n{listed}\n\n"
                'Return JSON like {"route": "<route name>", "reason": "..."}.\n'
                f"Question: {question}"
            ),
        },
    ]
