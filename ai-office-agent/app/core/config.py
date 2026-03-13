from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.path import get_project_root


class Settings(BaseSettings):
    APP_NAME: str = "AI Office Agent"
    APP_ENV: str = "development"
    APP_VERSION: str = "0.1.0"
    APP_DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    MODEL_PROVIDER: str = "dashscope"
    MODEL_NAME: str = "qwen3.5-plus"
    MODEL_ROUTING_ENABLED: bool = True
    MODEL_TURBO_NAME: str = "qwen-turbo"
    MODEL_PLUS_NAME: str = "qwen3.5-flash"
    MODEL_MAX_NAME: str = "qwen3.5-plus"
    MODEL_TEMPERATURE: float = 0.2
    MODEL_TIMEOUT: int = 60
    MODEL_MAX_RETRIES: int = 2
    API_KEY: str | None = Field(default=None, validation_alias="DASHSCOPE_API_KEY")
    BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    CHAT_HISTORY_LIMIT: int = 10
    TOOL_CALL_MAX_ITERATIONS: int = 5
    VECTOR_DB_PATH: str = ".chroma"
    CHROMA_COLLECTION_NAME: str = "ai_office_agent_docs"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    RAG_DEFAULT_TOP_K: int = 3
    RAG_BASE_RETRIEVAL_K: int = 10
    RAG_CHUNK_SIZE: int = 1000
    RAG_CHUNK_OVERLAP: int = 150
    RAG_RETRIEVAL_MULTIPLIER: int = 4
    RAG_RERANK_MODEL: str | None = None
    APP_TIMEZONE: str = "Asia/Shanghai"
    WEB_SEARCH_MAX_RESULTS: int = 5

    model_config = SettingsConfigDict(
        env_file=str(get_project_root() / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
