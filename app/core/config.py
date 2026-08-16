from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "platformpilot-rag"
    version: str = "0.1.0"
    database_url: str

    top_k: int = 4
    similarity_threshold: float = 0.5

    # None derives a bound from the visible CPUs. Set this explicitly in a
    # container: cpu_count() reports the node's cores, not the pod's limit.
    embedder_max_workers: int | None = None
    embedder_torch_threads: int = 1

    anthropic_api_key: str
    anthropic_model: str = "claude-sonnet-4-6"
    anthropic_timeout_seconds: float = 30.0
    max_context_tokens: int = 8000
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.0

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
