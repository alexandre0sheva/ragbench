"""Splitting and packing helpers shared by the structure-aware chunkers. Everything works on `(start, end)` character spans."""

from __future__ import annotations

import re
from bisect import bisect_left
from collections.abc import Sequence

from ragbench.documents.tokenizer import TokenCounter

Span = tuple[int, int]

SEPARATORS = ("\n\n", "\n", ". ", " ")

# A sentence ends at .!? followed by whitespace and something that opens a sentence, at CJK sentence punctuation, or at a blank line.
_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z0-9\"'“‘(\[])"
    r"|(?<=[.!?][\"”’)])\s+(?=[A-Z0-9\"'“‘(\[])"
    r"|(?<=[。！？])\s*"
    r"|\n[ \t]*\n\s*"
)


def sentence_spans(text: str) -> list[Span]:
    """Sentences that tile `text` (trailing whitespace belongs to the sentence before it)."""
    spans: list[Span] = []
    start = 0
    for match in _SENTENCE_BOUNDARY.finditer(text):
        if match.end() > start and match.end() < len(text):
            spans.append((start, match.end()))
            start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _pieces(text: str, start: int, end: int, separator: str) -> list[Span]:
    """Split `[start, end)` after every `separator`, keeping it attached to the piece before it."""
    pieces: list[Span] = []
    cursor = start
    while True:
        found = text.find(separator, cursor, end)
        if found < 0:
            break
        pieces.append((cursor, found + len(separator)))
        cursor = found + len(separator)
    if cursor < end:
        pieces.append((cursor, end))
    return pieces


def hard_split(start: int, end: int, size: int, count: TokenCounter) -> list[Span]:
    """Windows of `size` tokens that tile `[start, end)`; the last resort when no separator fits."""
    low, high = bisect_left(count._starts, start), bisect_left(count._starts, end)
    if high - low <= size:
        return [(start, end)]
    windows: list[Span] = []
    cursor = start
    for first in range(low + size, high, size):
        windows.append((cursor, count.spans[first][0]))
        cursor = count.spans[first][0]
    windows.append((cursor, end))
    return windows


def split_region(text: str, start: int, end: int, size: int, count: TokenCounter, separators: Sequence[str] = SEPARATORS) -> list[Span]:
    """Recursively split `[start, end)` into spans of at most `size` tokens, preferring the earliest separator that works."""
    if count(start, end) <= size:
        return [(start, end)]
    if not separators:
        return hard_split(start, end, size, count)
    pieces = _pieces(text, start, end, separators[0])
    if len(pieces) <= 1:
        return split_region(text, start, end, size, count, separators[1:])
    spans: list[Span] = []
    for piece in pieces:
        spans.extend(split_region(text, piece[0], piece[1], size, count, separators[1:]) if count(*piece) > size else [piece])
    return spans


def pack(segments: Sequence[Span], size: int, overlap: int, count: TokenCounter, *, unit: str = "tokens") -> list[Span]:
    """Greedily join consecutive `segments` into chunks of at most `size` tokens.

    `overlap` is measured in tokens (`unit="tokens"`) or in segments (`unit="units"`, e.g. sentences). Whole trailing
    segments carry over when they fit; with token overlap and no segment small enough, the tail of the last segment is
    carried instead, so a requested overlap is always delivered. A chunk never exceeds `size` because of overlap.
    """
    segs = list(segments)
    chunks: list[Span] = []
    i, n = 0, len(segs)
    while i < n:
        j = i + 1
        while j < n and count(segs[i][0], segs[j][1]) <= size:
            j += 1
        chunks.append((segs[i][0], segs[j - 1][1]))
        if j >= n:
            break
        k = j
        while overlap > 0 and k - 1 > i and _carried(segs, k - 1, j, count, unit) <= overlap and count(segs[k - 1][0], segs[j][1]) <= size:
            k -= 1
        if k == j and overlap > 0 and unit == "tokens":
            room = min(overlap, size - count(*segs[j]))
            begin = count.start_of_last(segs[j - 1][0], segs[j - 1][1], room) if room > 0 else None
            if begin is not None and begin > segs[j - 1][0]:
                segs[j - 1] = (begin, segs[j - 1][1])
                k = j - 1
        i = k
    return chunks


def _carried(segs: list[Span], first: int, stop: int, count: TokenCounter, unit: str) -> int:
    return stop - first if unit == "units" else count(segs[first][0], segs[stop - 1][1])


def merge_small(chunks: list[Span], minimum: int, size: int, count: TokenCounter) -> list[Span]:
    """Fold chunks below `minimum` tokens into a neighbour when the result still fits in `size`."""
    merged: list[Span] = []
    for chunk in chunks:
        if merged and count(*chunk) < minimum and count(merged[-1][0], chunk[1]) <= size:
            merged[-1] = (merged[-1][0], max(merged[-1][1], chunk[1]))
        else:
            merged.append(chunk)
    if len(merged) > 1 and count(*merged[0]) < minimum and count(merged[0][0], merged[1][1]) <= size:
        merged[1] = (merged[0][0], merged[1][1])
        merged.pop(0)
    return merged
