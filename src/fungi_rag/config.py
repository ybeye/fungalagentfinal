from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from fungi_rag.utils import ensure_dir


EmbeddingBackend = Literal["sentence_transformers", "hashing", "openai"]
GeneratorBackend = Literal["codex_bridge", "codex_cli", "transformers"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="FUNGI_",
        extra="ignore",
        case_sensitive=False,
    )

    embedding_backend: EmbeddingBackend = "sentence_transformers"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    generator_backend: GeneratorBackend = "codex_bridge"
    enable_codex_cli: bool = False
    hf_model: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    hf_adapter_path: Path | None = None
    hf_device: str = "auto"
    hf_max_new_tokens: int = Field(default=220, ge=20, le=1000)

    chroma_dir: Path = Path("data/chroma")
    upload_dir: Path = Path("data/uploads")
    background_dir: Path = Path("data/background")
    references_dir: Path = Path("data/references")
    source_raw_dir: Path = Path("data/sources/raw")
    source_state_path: Path = Path("data/sources/sources.jsonl")
    index_dir: Path = Path("data/index")
    output_dir: Path = Path("outputs")

    retrieval_top_k: int = Field(default=6, ge=1, le=30)
    chunk_size: int = Field(default=900, ge=200, le=4000)
    chunk_overlap: int = Field(default=120, ge=0, le=1000)

    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    langsmith_api_key: str | None = Field(default=None, validation_alias="LANGSMITH_API_KEY")

    def ensure_directories(self) -> None:
        for path in [
            self.chroma_dir,
            self.upload_dir,
            self.background_dir,
            self.references_dir,
            self.source_raw_dir,
            self.source_state_path.parent,
            self.index_dir,
            self.output_dir,
        ]:
            ensure_dir(path)


def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
