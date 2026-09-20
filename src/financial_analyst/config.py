from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    llm_model: str = Field(default="gpt-4.1-mini", alias="LLM_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    chroma_dir: Path = Field(default=Path("chroma_db"), alias="CHROMA_DIR")
    collection_name: str = Field(default="financial_documents", alias="COLLECTION_NAME")
    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    chunk_size: int = Field(default=1200, ge=200, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, ge=0, alias="CHUNK_OVERLAP")
    top_k: int = Field(default=6, ge=1, le=20, alias="TOP_K")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


@lru_cache
def get_settings() -> Settings:
    return Settings()
