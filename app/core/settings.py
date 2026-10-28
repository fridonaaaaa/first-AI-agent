from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_url: str = Field(
        "postgresql://postgres:postgres@localhost:5432/investigation_engine",
        description="PostgreSQL connection string with pgvector extension",
    )

    # Security
    admin_token: str = Field("dev-token-change-me", description="Token for admin endpoints")

    # LLM
    llm_provider: str = Field("mock", description="gemini | groq | claude | mock")
    llm_model: str = Field("", description="Model name; provider default used if empty")
    gemini_api_key: str = Field("", description="Google Gemini API key")
    groq_api_key: str = Field("", description="Groq API key")
    anthropic_api_key: str = Field("", description="Anthropic Claude API key")

    # Embeddings
    embedding_model: str = Field(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description="Must match the model used to embed the corpus. Never change after ingestion.",
    )
    embedding_dim: int = Field(384, description="Vector dimension; must match embedding_model")

    # Retrieval
    retrieval_mode: str = Field(
        "hybrid_rerank",
        description="dense_only | bm25_only | hybrid | hybrid_rerank",
    )
    retrieval_top_k: int = Field(10, ge=1, le=100)
    bm25_k1: float = Field(1.5, gt=0)
    bm25_b: float = Field(0.75, ge=0.0, le=1.0)
    rrf_k: int = Field(60, gt=0)

    # RAG pipeline
    max_tool_steps: int = Field(8, ge=1, le=20)
    max_sub_queries: int = Field(4, ge=1, le=8)
    # Minimum cosine similarity for the top retrieved chunk to pass the pre-LLM
    # abstention gate.  RRF scores are bounded below by 1/(k+top_k) ≈ 0.014 and
    # cannot signal "irrelevant", so we use raw cosine similarity instead.
    abstention_threshold: float = Field(0.25, ge=0.0, le=1.0)

    # Rate limiting
    rate_limit_per_ip_per_minute: int = Field(20, ge=1)
    daily_cap_ask: int = Field(1000, ge=1)

    # Caching
    llm_cache_dir: str = Field("/tmp/llm_cache")

    # Misc
    log_level: str = Field("INFO")
    environment: str = Field("development")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
