class InvestigationEngineError(Exception):
    """Base for all application errors."""


class DatabaseError(InvestigationEngineError):
    """A database operation failed."""


class EmbeddingError(InvestigationEngineError):
    """Embedding generation failed."""


class LLMError(InvestigationEngineError):
    """An LLM call failed."""


class LLMRateLimitError(LLMError):
    """LLM provider returned 429 Too Many Requests."""


class LLMTimeoutError(LLMError):
    """LLM call exceeded the configured timeout."""


class LLMMalformedResponseError(LLMError):
    """LLM returned JSON that failed schema validation after one repair attempt."""


class DocumentNotFoundError(InvestigationEngineError):
    """A requested document ID is not in the database."""


class DuplicateDocumentError(InvestigationEngineError):
    """An exact-duplicate document was uploaded (same normalized content hash)."""


class AuthorizationError(InvestigationEngineError):
    """An admin-only request is missing or has an invalid token."""


class RateLimitError(InvestigationEngineError):
    """A client exceeded the configured rate limit."""


class IngestionError(InvestigationEngineError):
    """Document parsing or chunking failed."""

    def __init__(
        self,
        message: str,
        source: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.source = source
        self.cause = cause
