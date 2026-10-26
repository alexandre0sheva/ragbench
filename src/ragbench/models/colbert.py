"""Per-token (late-interaction) encoders for `late_interaction`: a ColBERT checkpoint, or deterministic hashing vectors in mock runs.

The real encoder is a small in-house loader (Hugging Face encoder + the checkpoint's projection layer + the `[Q]` / `[D]` markers and query
padding), because PyLate pins an older sentence-transformers than RAGBench's floor. It reads the two checkpoint layouts in use: Stanford
ColBERT (`artifact.metadata`, the projection inside `model.safetensors`) and PyLate (`config_sentence_transformers.json`, `1_Dense/`). It needs
the `colbert` extra; one model is loaded per run and shared by every system that names it.
"""

from __future__ import annotations

import hashlib
import json
import string
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ragbench.utils.text import tokenize

DEFAULT_COLBERT_MODEL = "answerdotai/answerai-colbert-small-v1"
HASHING_DIM = 64
BATCH_SIZE = 32
_CHECKPOINT_FILES = ["*.json", "*.txt", "*.model", "artifact.metadata", "model.safetensors", "1_Dense/*"]


def maxsim(query: np.ndarray, document: np.ndarray) -> float:
    """Late-interaction score: for each query vector its best dot product with any document vector, summed."""
    if len(query) == 0 or len(document) == 0:
        return 0.0
    return float((query.astype(np.float32) @ document.astype(np.float32).T).max(axis=1).sum())


class TokenEncoder(ABC):
    name: str

    @abstractmethod
    def encode_query(self, text: str) -> np.ndarray:
        """One L2-normalised vector per query token (padding included where the model expands queries)."""

    @abstractmethod
    def prepare_documents(self, texts: list[str]) -> list[Any]:
        """Tokenise documents once; `vector_count` and `encode_prepared` both work from the result."""

    @abstractmethod
    def vector_count(self, prepared: Any) -> int:
        """How many vectors one prepared document will have (for the index-size guard, before anything is encoded)."""

    @abstractmethod
    def encode_prepared(self, prepared: list[Any]) -> list[np.ndarray]:
        """Per prepared document, one L2-normalised vector per kept token."""

    def encode_documents(self, texts: list[str]) -> list[np.ndarray]:
        return self.encode_prepared(self.prepare_documents(texts))

    def document_vector_counts(self, texts: list[str]) -> list[int]:
        return [self.vector_count(prepared) for prepared in self.prepare_documents(texts)]


class HashingTokenEncoder(TokenEncoder):
    """Offline stand-in: each word gets a fixed random unit vector (seeded by its hash), so equal words match exactly and MaxSim runs for real."""

    name = "hashing-colbert"

    def encode_query(self, text: str) -> np.ndarray:
        return _hash_vectors(tuple(tokenize(text)))

    def prepare_documents(self, texts: list[str]) -> list[Any]:
        return [tuple(tokenize(text)) for text in texts]

    def vector_count(self, prepared: Any) -> int:
        return len(prepared)

    def encode_prepared(self, prepared: list[Any]) -> list[np.ndarray]:
        return [_hash_vectors(tokens) for tokens in prepared]


def _hash_vectors(tokens: tuple[str, ...]) -> np.ndarray:
    return np.vstack([_token_vector(token) for token in tokens]) if tokens else np.zeros((0, HASHING_DIM), dtype=np.float32)


@lru_cache(maxsize=65536)
def _token_vector(token: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "little")
    vector = np.random.default_rng(seed).normal(size=HASHING_DIM).astype(np.float32)
    return vector / np.linalg.norm(vector)


# --- checkpoints ------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ColBERTSettings:
    query_marker: str  # token inserted after the first token of a query
    document_marker: str
    query_length: int  # queries are padded (expanded) to this many tokens
    document_length: int  # documents are cut to this many tokens
    attend_to_expansion: bool
    skiplist: tuple[str, ...]  # tokens whose vectors a document does not keep (punctuation)
    projection_file: str  # holds `linear.weight`
    query_prompt: str = ""  # text put before a query (PyLate prompts)
    document_prompt: str = ""


def read_settings(directory: Path) -> ColBERTSettings:
    """The encoding settings of a checkpoint folder: PyLate (`config_sentence_transformers.json` and `1_Dense/`) or Stanford ColBERT (`artifact.metadata`)."""
    pylate = directory / "config_sentence_transformers.json"
    if pylate.exists() and (directory / "1_Dense").is_dir():
        config = json.loads(pylate.read_text(encoding="utf-8"))
        prompts = config.get("prompts") or {}
        return ColBERTSettings(
            query_marker=str(config.get("query_prefix", "[Q] ")).strip(),
            document_marker=str(config.get("document_prefix", "[D] ")).strip(),
            query_length=int(config.get("query_length", 32)),
            document_length=int(config.get("document_length", 180)),
            attend_to_expansion=bool(config.get("attend_to_expansion_tokens", False)),
            skiplist=tuple(config.get("skiplist_words") or ()),
            projection_file="1_Dense/model.safetensors",
            query_prompt=str(prompts.get("query", "")),
            document_prompt=str(prompts.get("document", "")),
        )
    stanford = directory / "artifact.metadata"
    if stanford.exists():
        meta = json.loads(stanford.read_text(encoding="utf-8"))
        return ColBERTSettings(
            query_marker=str(meta.get("query_token_id", "[unused0]")),
            document_marker=str(meta.get("doc_token_id", "[unused1]")),
            query_length=int(meta.get("query_maxlen", 32)),
            document_length=int(meta.get("doc_maxlen", 180)),
            attend_to_expansion=bool(meta.get("attend_to_mask_tokens", False)),
            skiplist=tuple(string.punctuation) if meta.get("mask_punctuation", True) else (),
            projection_file="model.safetensors",
        )
    raise ValueError(f"{directory} is not a ColBERT checkpoint this loader reads (no PyLate config_sentence_transformers.json + 1_Dense/, no Stanford artifact.metadata)")


def frame_query(ids: list[int], *, marker_id: int, length: int, mask_id: int, attend_to_expansion: bool) -> tuple[list[int], list[int]]:
    """A tokenised query (first and last special tokens included) with its marker after the first token, cut or padded with mask tokens to
    `length`; the mask tokens are attended to only when the model was trained that way. Returns the ids and the attention mask."""
    framed = _with_marker(ids, marker_id, length)
    attention = [1] * len(framed)
    padding = length - len(framed)
    return framed + [mask_id] * padding, attention + [1 if attend_to_expansion else 0] * padding


def frame_document(ids: list[int], *, marker_id: int, length: int, skip_ids: set[int]) -> tuple[list[int], list[bool]]:
    """A tokenised document with its marker after the first token, cut to `length` (the last token kept); and which positions keep a vector."""
    framed = _with_marker(ids, marker_id, length)
    return framed, [token not in skip_ids for token in framed]


def marker_token_id(tokenizer: Any, token: str) -> int:
    """The id of a marker token; a marker missing from the vocabulary is an error, not the unknown token (which would skew every vector)."""
    found = tokenizer.convert_tokens_to_ids(token)
    if not isinstance(found, int) or found == tokenizer.unk_token_id:
        raise ValueError(f"the checkpoint's tokenizer has no {token!r} token; this ColBERT checkpoint does not match the layout it was read as")
    return found


def _with_marker(ids: list[int], marker_id: int, length: int) -> list[int]:
    body = ids[1:-1][: max(0, length - 3)]
    return [ids[0], marker_id, *body, ids[-1]]


class ColBERTEncoder(TokenEncoder):
    """A ColBERT checkpoint run with `transformers`: encoder output, projected by the checkpoint's linear layer, L2-normalised."""

    def __init__(self, name: str):
        try:
            import torch
            from huggingface_hub import snapshot_download
            from safetensors.torch import load_file
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(f"late_interaction needs the colbert extra for {name!r}: pip install 'ragbench[colbert]' ({exc})") from exc
        self.name = name
        directory = Path(name) if Path(name).is_dir() else Path(snapshot_download(name, allow_patterns=_CHECKPOINT_FILES))
        self.settings = read_settings(directory)
        self.tokenizer = AutoTokenizer.from_pretrained(directory)
        self.model = AutoModel.from_pretrained(directory).eval()
        weights = load_file(str(directory / self.settings.projection_file))
        key = next(k for k in weights if k.endswith("linear.weight"))
        self.projection = weights[key].float()
        self.torch = torch
        self.query_marker_id = marker_token_id(self.tokenizer, self.settings.query_marker)
        self.document_marker_id = marker_token_id(self.tokenizer, self.settings.document_marker)
        self.mask_id = self.tokenizer.mask_token_id if self.tokenizer.mask_token_id is not None else self.tokenizer.pad_token_id
        self.skip_ids = {i for word in self.settings.skiplist for i in self.tokenizer.encode(word, add_special_tokens=False)}
        self._lock = threading.Lock()  # one forward pass at a time: the model is shared by every system and question thread

    def _ids(self, texts: list[str], prompt: str, length: int) -> list[list[int]]:
        """Token ids with the special tokens, cut by the tokenizer to `length` (framing then makes room for the marker)."""
        return [self.tokenizer.encode(prompt + text, add_special_tokens=True, truncation=True, max_length=length) for text in texts]

    def _forward(self, batch: list[list[int]], attention: list[list[int]]) -> Any:
        width = max(len(ids) for ids in batch)
        pad = self.tokenizer.pad_token_id or 0
        input_ids = self.torch.tensor([ids + [pad] * (width - len(ids)) for ids in batch])
        mask = self.torch.tensor([att + [0] * (width - len(att)) for att in attention])
        with self._lock, self.torch.no_grad():
            hidden = self.model(input_ids=input_ids, attention_mask=mask).last_hidden_state
            projected = hidden @ self.projection.T
            return self.torch.nn.functional.normalize(projected, dim=-1).numpy()

    def encode_query(self, text: str) -> np.ndarray:
        s = self.settings
        ids, attention = frame_query(self._ids([text], s.query_prompt, s.query_length)[0], marker_id=self.query_marker_id, length=s.query_length, mask_id=self.mask_id, attend_to_expansion=s.attend_to_expansion)
        return self._forward([ids], [attention])[0].astype(np.float32)

    def prepare_documents(self, texts: list[str]) -> list[Any]:
        s = self.settings
        return [frame_document(ids, marker_id=self.document_marker_id, length=s.document_length, skip_ids=self.skip_ids) for ids in self._ids(texts, s.document_prompt, s.document_length)]

    def vector_count(self, prepared: Any) -> int:
        return sum(prepared[1])

    def encode_prepared(self, prepared: list[Any]) -> list[np.ndarray]:
        vectors: list[np.ndarray] = []
        for start in range(0, len(prepared), BATCH_SIZE):
            batch = prepared[start : start + BATCH_SIZE]
            out = self._forward([ids for ids, _ in batch], [[1] * len(ids) for ids, _ in batch])
            vectors.extend(out[row, : len(ids)][np.array(keep)].astype(np.float16) for row, (ids, keep) in enumerate(batch))
        return vectors



_ENCODERS: dict[str, TokenEncoder] = {}
_ENCODERS_LOCK = threading.Lock()


def load_token_encoder(name: str, *, force_mock: bool) -> TokenEncoder:
    """The encoder for `name`, loaded once per process and shared; mock runs get the hashing encoder and never load (or download) a model."""
    if force_mock:
        return HashingTokenEncoder()
    with _ENCODERS_LOCK:
        if name not in _ENCODERS:
            _ENCODERS[name] = ColBERTEncoder(name)
        return _ENCODERS[name]
