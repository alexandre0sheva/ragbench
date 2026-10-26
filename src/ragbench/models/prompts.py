"""Phrases that identify LLM prompts of ingestion and query planning. The systems write them into their prompts and `MockLLM` recognises them, so
both sides import them from here instead of repeating the text."""

from __future__ import annotations

SITUATE_CHUNK_MARKER = "situate this chunk within the overall document"
SUMMARIZE_DOCUMENT_MARKER = "Write a short summary of this document so it can be found later."

# `graph_rag`: entity/relation extraction and community reports at ingestion (the first JSON, the second plain text), and the
# map step of its `global` search (JSON).
EXTRACT_GRAPH_MARKER = "Extract the named entities in the text below and the relations between them"
SUMMARIZE_COMMUNITY_MARKER = "Write a report on the community of related entities below"
MAP_COMMUNITY_MARKER = "List the points of the community report below that help answer the question"

# `compression`'s `llm_extract` compressor (JSON) and `raptor`'s cluster summaries at ingestion (plain text).
COMPRESS_PASSAGES_MARKER = "Quote, word for word, the sentences of each numbered passage below that help answer the question"
SUMMARIZE_CLUSTER_MARKER = "Write a summary of the related passages below"

# Query-time planning prompts of `rag_fusion` and `decompose` (JSON answers).
GENERATE_QUERY_VARIANTS_MARKER = "Write alternative search queries for the question below"
PLAN_SUBQUESTIONS_MARKER = "Break the question below into the ordered sub-questions needed to answer it"

# Control prompts of the agentic systems (`corrective`, `iterative`); the text around them lives in `agents/prompts.py`.
GRADE_CHUNKS_MARKER = "Grade how relevant each numbered passage is to the question"
REWRITE_QUERY_MARKER = "Write a better search query that finds the missing information"
NEXT_HOP_MARKER = "Decide what to do next to answer the question from the evidence so far"
CHECK_GROUNDED_MARKER = "Check whether the answer below is fully supported by the context"

# Tool-using agents (`agent_search`, `grep_agent`): the system prompt carries the first marker; ReAct-JSON mode adds the second; the last
# introduces the "answer now" turn sent when the budget ran out.
TOOL_AGENT_MARKER = "You are a research agent with tools"
REACT_JSON_MARKER = "Reply with exactly one JSON object"
FORCE_ANSWER_MARKER = "You cannot use any more tools"

# The adaptive system's LLM router.
ROUTE_QUESTION_MARKER = "Choose the route that should handle the question below"

# Dataset synthesis and pooled labeling (`ragbench generate-questions`, `ragbench label`); the text around them lives in `datasets/prompts.py`.
WRITE_QUESTION_MARKER = "Write one question that the document below answers"
WRITE_MULTIHOP_MARKER = "Write one question that needs both documents below to answer"
PARAPHRASE_QUESTION_MARKER = "Rewrite the question below so that it avoids the words"
WRITE_UNANSWERABLE_MARKER = "Write one question that none of the documents below answers"
GRADE_DOCUMENT_MARKER = "Grade how well the document below answers the question"
