"""The OpenRouter model catalog: prices, tool / JSON support and reasoning switches of every `openrouter:` model.

`GET https://openrouter.ai/api/v1/models` (public, no key) is trimmed to what RAGBench needs and saved as a dated JSON file by
`ragbench models sync`. Everything else reads that file, offline: the price table (`models/cost.py`), the `openrouter:` provider
(which parameters a model accepts) and `ragbench models list`. Until the first sync a snapshot of the shortlisted models in
docs/models.md, bundled with the package, is used. The newer of the two files wins.

Prices are USD per 1M tokens like the rest of RAGBench (the API quotes USD per token). A prompt-length override
(`min_prompt_tokens: N`) becomes a price tier for prompts above N tokens; time-of-day discounts are ignored (the base price is
kept, an upper bound). A model whose price is variable (a router, quoted as a negative number) gets no price.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

MODELS_URL = "https://openrouter.ai/api/v1/models"
CATALOG_ENV = "RAGBENCH_OPENROUTER_CATALOG"
BUNDLED_CATALOG = Path(__file__).parent / "data" / "openrouter_models.json"
# Reasoning efforts from least to most thinking; the default is the first one a model accepts.
EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
_KEPT_PARAMETERS = ("tools", "tool_choice", "response_format", "structured_outputs", "reasoning", "temperature", "max_tokens")


@dataclass(frozen=True)
class OpenRouterModel:
    id: str
    name: str = ""
    hugging_face_id: str | None = None
    context_length: int | None = None
    price: dict[str, Any] | None = None  # a `models/cost.py` price row, or None when the price is variable
    supported_parameters: tuple[str, ...] = ()
    reasoning_mandatory: bool = False
    supported_efforts: tuple[str, ...] = ()

    @property
    def supports_tools(self) -> bool:
        return "tools" in self.supported_parameters

    @property
    def supports_json(self) -> bool:
        return "response_format" in self.supported_parameters or "structured_outputs" in self.supported_parameters

    @property
    def open_weights(self) -> bool:
        return bool(self.hugging_face_id)

    def lowest_effort(self) -> str | None:
        """The least reasoning the model accepts: `none` when it can switch reasoning off, else its lowest listed effort."""
        if not self.supported_efforts:
            return None if self.reasoning_mandatory else "none"
        return min(self.supported_efforts, key=lambda effort: EFFORT_ORDER.index(effort) if effort in EFFORT_ORDER else len(EFFORT_ORDER))


@dataclass
class Catalog:
    fetched_at: str
    source: str
    models: dict[str, OpenRouterModel] = field(default_factory=dict)
    path: Path | None = None


def _per_million(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number < 0 else round(number * 1_000_000, 6)


def _price(pricing: dict[str, Any]) -> dict[str, Any] | None:
    base_in, base_out = _per_million(pricing.get("prompt")), _per_million(pricing.get("completion"))
    if base_in is None or base_out is None:
        return None
    row: dict[str, Any] = {"input": base_in, "output": base_out}
    above = sorted((o for o in pricing.get("overrides") or [] if isinstance(o, dict) and o.get("min_prompt_tokens")), key=lambda o: int(o["min_prompt_tokens"]))
    tiers: list[dict[str, Any]] = []
    current = {"input": base_in, "output": base_out}
    for override in above:
        rate_in, rate_out = _per_million(override.get("prompt", pricing.get("prompt"))), _per_million(override.get("completion", pricing.get("completion")))
        if rate_in is None or rate_out is None:
            continue
        tiers.append({**current, "max_prompt_tokens": int(override["min_prompt_tokens"])})
        current = {"input": rate_in, "output": rate_out}
    if tiers:
        row["tiers"] = [*tiers, current]
    return row


def trim(api_model: dict[str, Any]) -> dict[str, Any]:
    """One entry of the API's `data` list, reduced to the fields the catalog file keeps."""
    reasoning = api_model.get("reasoning") or {}
    return {
        "name": api_model.get("name") or "",
        "hugging_face_id": api_model.get("hugging_face_id") or None,
        "context_length": api_model.get("context_length"),
        "price": _price(api_model.get("pricing") or {}),
        "supported_parameters": [p for p in api_model.get("supported_parameters") or [] if p in _KEPT_PARAMETERS],
        "reasoning": {"mandatory": bool(reasoning.get("mandatory")), "supported_efforts": list(reasoning.get("supported_efforts") or [])},
    }


def catalog_payload(api_response: dict[str, Any], fetched_at: str, only: set[str] | None = None) -> dict[str, Any]:
    """The catalog file's content for a `GET /api/v1/models` response (optionally only the model ids in `only`)."""
    data = api_response.get("data")
    if not isinstance(data, list):
        raise ValueError("not an OpenRouter /api/v1/models response: it has no `data` list")
    models = {str(m["id"]): trim(m) for m in data if isinstance(m, dict) and m.get("id") and (only is None or m["id"] in only)}
    return {"source": MODELS_URL, "fetched_at": fetched_at, "models": dict(sorted(models.items()))}


def _model(model_id: str, entry: dict[str, Any]) -> OpenRouterModel:
    reasoning = entry.get("reasoning") or {}
    return OpenRouterModel(
        id=model_id,
        name=str(entry.get("name") or ""),
        hugging_face_id=entry.get("hugging_face_id"),
        context_length=entry.get("context_length"),
        price=entry.get("price"),
        supported_parameters=tuple(entry.get("supported_parameters") or ()),
        reasoning_mandatory=bool(reasoning.get("mandatory")),
        supported_efforts=tuple(reasoning.get("supported_efforts") or ()),
    )


def read_catalog(path: Path) -> Catalog:
    payload = json.loads(path.read_text(encoding="utf-8"))
    models = {model_id: _model(model_id, entry) for model_id, entry in (payload.get("models") or {}).items()}
    return Catalog(fetched_at=str(payload.get("fetched_at") or ""), source=str(payload.get("source") or MODELS_URL), models=models, path=path)


def synced_catalog_path() -> Path:
    """Where `ragbench models sync` writes: `$RAGBENCH_OPENROUTER_CATALOG`, else the user cache directory."""
    explicit = os.environ.get(CATALOG_ENV)
    if explicit:
        return Path(explicit).expanduser()
    cache_home = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return cache_home / "ragbench" / "openrouter_models.json"


_lock = threading.Lock()
_loaded: dict[Path, tuple[float, Catalog]] = {}


def _cached_read(path: Path) -> Catalog | None:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    hit = _loaded.get(path)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    try:
        catalog = read_catalog(path)
    except (OSError, ValueError):
        return None
    _loaded[path] = (mtime, catalog)
    return catalog


def load_catalog() -> Catalog:
    """The newer of the synced catalog and the bundled snapshot (re-read when the synced file changes)."""
    with _lock:
        candidates = [c for c in (_cached_read(synced_catalog_path()), _cached_read(BUNDLED_CATALOG)) if c is not None]
    if not candidates:
        return Catalog(fetched_at="", source=MODELS_URL)
    return max(candidates, key=lambda c: c.fetched_at)


def model_info(model_id: str) -> OpenRouterModel | None:
    return load_catalog().models.get(model_id)


def fetch_models(timeout: float = 30.0) -> dict[str, Any]:
    """`GET /api/v1/models` (public; no key is sent)."""
    import httpx

    response = httpx.get(MODELS_URL, timeout=timeout, headers={"accept": "application/json"})
    response.raise_for_status()
    return response.json()


def write_catalog(api_response: dict[str, Any], path: Path | None = None, today: date | None = None) -> Catalog:
    """Save `api_response` as the dated catalog file at `path` (default: `synced_catalog_path()`) and return it."""
    target = path or synced_catalog_path()
    payload = catalog_payload(api_response, (today or date.today()).isoformat())
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    tmp.replace(target)
    return read_catalog(target)
