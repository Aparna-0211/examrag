"""Central settings, read from environment variables (or a .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass

try:  # python-dotenv is optional at import time so tests stay lightweight
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "anthropic"  # "anthropic" | "ollama"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    embedding_model: str = "BAAI/bge-m3"
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 5
    min_similarity: float = 0.35
    chroma_dir: str = "data/chroma"


def load_settings() -> Settings:
    """Build Settings from the current environment."""
    env = os.environ.get
    return Settings(
        llm_provider=env("LLM_PROVIDER", "anthropic").lower(),
        anthropic_api_key=env("ANTHROPIC_API_KEY", ""),
        anthropic_model=env("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
        ollama_url=env("OLLAMA_URL", "http://localhost:11434"),
        ollama_model=env("OLLAMA_MODEL", "qwen2.5:7b"),
        embedding_model=env("EMBEDDING_MODEL", "BAAI/bge-m3"),
        chunk_size=int(env("CHUNK_SIZE", "900")),
        chunk_overlap=int(env("CHUNK_OVERLAP", "150")),
        top_k=int(env("TOP_K", "5")),
        min_similarity=float(env("MIN_SIMILARITY", "0.35")),
        chroma_dir=env("CHROMA_DIR", "data/chroma"),
    )
