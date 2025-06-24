

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    # Environment
    ENV: str = Field("development", description="Environment name: development | staging | production")
    DEBUG: bool = Field(True, description="Toggle debug mode for logging and features")

    # General LLM model (used for GPTService or fallback generation)
    MODEL_NAME: str = Field("gpt-3.5-turbo", description="OpenAI model used by GPTService")

    # RAG-specific settings
    EMBED_MODEL: str = Field("sentence-transformers/all-MiniLM-L6-v2", description="Embedding model for FAISS")
    LLM_REPO: str = Field("google/flan-t5-base", description="HuggingFace LLM repo for RAG generation")
    TOP_K: int = Field(3, description="Number of top documents to retrieve in RAG")
    FAISS_INDEX_PATH: str = Field("data/faiss_index", description="Path to local FAISS index")
    FAISS_ALLOW_UNSAFE_LOAD: bool = Field(True, description="Allow unsafe deserialization for FAISS index")

    # Document Ingestion
    DOCS_PATH: str = Field("data/source_pdfs", description="Path to source PDFs for ingestion")
    CHUNK_SIZE: int = Field(1000, description="Chunk size for document splitting")
    CHUNK_OVERLAP: int = Field(200, description="Chunk overlap for text splitting")

    # Secrets (auto-loaded from .env or system environment)
    OPENAI_API_KEY: str = Field(..., repr=False, description="API key for OpenAI GPT")
    HUGGINGFACEHUB_API_TOKEN: str = Field(..., repr=False, description="API key for HuggingFace models")

    # Settings config (loads from .env two levels up)
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parents[1] / ".env"),
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
