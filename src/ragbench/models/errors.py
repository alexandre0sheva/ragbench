"""Error taxonomy for model/provider calls.

Retry policy hangs off these types: `RateLimitError` and `TransientModelError` are
retried with backoff, everything else (`PermanentModelError`, `ModelInitError`) fails fast.
"""

from __future__ import annotations

from typing import Any


class RagbenchModelError(Exception):
    """Base class for errors raised by LLM, embedding, and reranker providers."""


class RateLimitError(RagbenchModelError):
    """Provider asked us to slow down (HTTP 429). `retry_after` is seconds, when the provider said so."""

    def __init__(self, message: str = "rate limited", retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class TransientModelError(RagbenchModelError):
    """Server errors, timeouts, and connection failures: worth retrying."""


class PermanentModelError(RagbenchModelError):
    """Auth failures, bad requests, unknown models: retrying cannot help."""


class ModelInitError(RagbenchModelError):
    """A real (non-mock) client could not be constructed even though credentials are present."""


class MissingExtraError(ImportError):
    """An optional SDK is not installed; the message names the extra (`pip install 'ragbench[anthropic]'`) that provides it."""


def _retry_after_seconds(exc: Any) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _translate_sdk_exception(exc: Exception, sdk: Any) -> Exception:
    """Shared by SDKs with the openai/anthropic exception layout (`RateLimitError`, `APIStatusError`, ...)."""
    if isinstance(exc, RagbenchModelError):
        return exc
    message = str(exc)
    if isinstance(exc, sdk.RateLimitError):
        error: RagbenchModelError = RateLimitError(message, retry_after=_retry_after_seconds(exc))
    elif isinstance(exc, sdk.APITimeoutError | sdk.APIConnectionError | sdk.InternalServerError):
        error = TransientModelError(message)
    elif isinstance(exc, sdk.APIStatusError):
        error = TransientModelError(message) if exc.status_code in (408, 409) or exc.status_code >= 500 else PermanentModelError(message)
    else:
        return exc
    error.__cause__ = exc
    return error


def translate_openai_exception(exc: Exception) -> Exception:
    """Map an OpenAI SDK exception onto the taxonomy; anything else is returned unchanged."""
    try:
        import openai
    except ImportError:  # pragma: no cover - openai is a core dependency
        return exc
    return _translate_sdk_exception(exc, openai)


def translate_anthropic_exception(exc: Exception) -> Exception:
    """Map an Anthropic SDK exception (HTTP 529 "overloaded" included) onto the taxonomy; anything else is returned unchanged."""
    try:
        import anthropic
    except ImportError:
        return exc  # the optional extra is absent, so no SDK exception can have been raised
    return _translate_sdk_exception(exc, anthropic)
