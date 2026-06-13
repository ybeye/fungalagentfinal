from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from fungi_rag.utils import stable_id, utc_now_iso


SourceStrategy = Literal["local", "seed", "hybrid"]
SafetyMode = Literal["strict", "academic"]


class ResearchBrief(BaseModel):
    title: str = Field(min_length=3)
    topic: str = Field(min_length=2)
    learning_objectives: list[str] = Field(default_factory=list, min_length=1)
    audience: str = Field(default="Academic learners")
    section_names: list[str] = Field(default_factory=list, min_length=1)
    number_of_paragraphs: int | dict[str, int] = 2
    source_strategy: SourceStrategy = "seed"
    seed_references: list[str] = Field(default_factory=list)
    number_of_queries: int = Field(default=5, ge=1, le=30)
    max_revisions: int = Field(default=2, ge=0, le=10)
    temperature: float = Field(default=0.2, ge=0.0, le=1.5)
    safety_mode: SafetyMode = "strict"

    @field_validator("topic")
    @classmethod
    def normalize_topic(cls, value: str) -> str:
        normalized = value.strip()
        if "funghi" in normalized.lower():
            normalized = normalized.lower().replace("funghi", "fungi")
        return normalized

    @field_validator("learning_objectives", "section_names")
    @classmethod
    def strip_list_values(cls, values: list[str]) -> list[str]:
        cleaned = []
        for value in values:
            if not value:
                continue
            value = value.strip()
            if value:
                cleaned.append(value)
        if not cleaned:
            raise ValueError("at least one non-empty value is required")
        return cleaned

    @model_validator(mode="after")
    def validate_paragraphs(self) -> "ResearchBrief":
        if isinstance(self.number_of_paragraphs, int):
            if self.number_of_paragraphs < 1:
                raise ValueError("number_of_paragraphs must be positive")
            return self
        missing = []
        for section in self.section_names:
            if section not in self.number_of_paragraphs:
                missing.append(section)
        if missing:
            raise ValueError(f"paragraph counts missing for sections: {missing}")
        for section, count in self.number_of_paragraphs.items():
            if count < 1:
                raise ValueError(f"paragraph count for {section!r} must be positive")
        return self

    def paragraph_plan(self) -> dict[str, int]:
        if isinstance(self.number_of_paragraphs, int):
            return {section: self.number_of_paragraphs for section in self.section_names}
        return dict(self.number_of_paragraphs)


class SourceManifestEntry(BaseModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_.-]+$")
    title: str
    url: str
    source_type: Literal["html", "pdf", "text", "markdown"] = "html"
    license_note: str = "Open/public source"
    topics: list[str] = Field(default_factory=list)
    corpus_role: Literal["background", "reference"] | None = None


class SourceManifest(BaseModel):
    sources: list[SourceManifestEntry] = Field(default_factory=list, min_length=1)

    @field_validator("sources")
    @classmethod
    def source_ids_unique(cls, values: list[SourceManifestEntry]) -> list[SourceManifestEntry]:
        seen = set()
        for value in values:
            if value.id in seen:
                raise ValueError("source ids must be unique")
            seen.add(value.id)
        return values


class SourceDocument(BaseModel):
    source_id: str
    title: str
    canonical_url: str | None = None
    local_path: str | None = None
    source_type: str = "text"
    license_note: str = "Unknown"
    content_hash: str
    retrieved_at: str = Field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceChunk(BaseModel):
    chunk_id: str
    source_id: str
    title: str
    text: str
    chunk_index: int
    content_hash: str
    canonical_url: str | None = None
    local_path: str | None = None
    section: str | None = None
    page: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_text(
        cls,
        *,
        source: SourceDocument,
        text: str,
        chunk_index: int,
        section: str | None = None,
        page: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "SourceChunk":
        chunk_hash = stable_id(source.source_id, chunk_index, text, length=24)
        return cls(
            chunk_id=f"{source.source_id}:{chunk_index}:{chunk_hash[:8]}",
            source_id=source.source_id,
            title=source.title,
            text=text,
            chunk_index=chunk_index,
            content_hash=chunk_hash,
            canonical_url=source.canonical_url,
            local_path=source.local_path,
            section=section,
            page=page,
            metadata=metadata or {},
        )


class Citation(BaseModel):
    citation_id: int
    source_id: str
    chunk_id: str
    title: str
    url: str | None = None
    path: str | None = None
    snippet: str


class EvidenceItem(BaseModel):
    citation_id: int
    chunk_id: str
    source_id: str
    title: str
    snippet: str
    url: str | None = None
    path: str | None = None
    vector_score: float = 0.0
    keyword_score: float = 0.0
    fused_score: float = 0.0
    confidence_note: str = "Retrieved by hybrid rank fusion."
    metadata: dict[str, Any] = Field(default_factory=dict)

    def citation(self) -> Citation:
        return Citation(
            citation_id=self.citation_id,
            source_id=self.source_id,
            chunk_id=self.chunk_id,
            title=self.title,
            url=self.url,
            path=self.path,
            snippet=self.snippet,
        )


class EvidencePacket(BaseModel):
    query: str
    normalized_query: str
    items: list[EvidenceItem]
    generated_at: str = Field(default_factory=utc_now_iso)
    safety_notes: list[str] = Field(default_factory=list)


class RagTrace(BaseModel):
    run_id: str
    query: str
    normalized_query: str
    vector_candidates: list[dict[str, Any]] = Field(default_factory=list)
    keyword_candidates: list[dict[str, Any]] = Field(default_factory=list)
    fused_candidates: list[dict[str, Any]] = Field(default_factory=list)
    selected_evidence: list[dict[str, Any]] = Field(default_factory=list)
    rejected_duplicates: list[dict[str, Any]] = Field(default_factory=list)
    generator_packet_path: str | None = None
    citation_audit: dict[str, Any] = Field(default_factory=dict)


class GenerationRequest(BaseModel):
    run_id: str
    step: str
    task: str
    evidence: EvidencePacket
    output_dir: Path
    response_schema: dict[str, Any] = Field(default_factory=dict)
    safety_mode: SafetyMode = "strict"


class GenerationResult(BaseModel):
    status: Literal["pending", "accepted", "invalid", "failed"]
    step: str
    prompt_path: str
    evidence_path: str
    schema_path: str
    response_path: str
    text: str = ""
    validation_errors: list[str] = Field(default_factory=list)


class AgentState(BaseModel):
    run_id: str
    brief: ResearchBrief | None = None
    queries: list[str] = Field(default_factory=list)
    evidence_packets: list[EvidencePacket] = Field(default_factory=list)
    outline: str = ""
    draft: str = ""
    citations: list[Citation] = Field(default_factory=list)
    output_dir: str = ""
    status: str = "initialized"


class EvaluationCase(BaseModel):
    id: str
    query: str
    expected_source_terms: list[str] = Field(default_factory=list)
    expected_answer_terms: list[str] = Field(default_factory=list)
    should_refuse: bool = False


def load_brief_yaml(path: Path) -> ResearchBrief:
    import yaml

    return ResearchBrief.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
