from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from fungi_rag.config import Settings, get_settings
from fungi_rag.embeddings import EmbeddingBackend, build_embedding_backend
from fungi_rag.models import EvidenceItem, EvidencePacket, RagTrace, SourceChunk
from fungi_rag.utils import (
    append_jsonl,
    ensure_dir,
    normalize_whitespace,
    read_jsonl,
    stable_id,
    write_json,
)


TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]+")


def normalize_query(query: str) -> str:
    normalized = normalize_whitespace(query.lower())
    return normalized.replace("funghi", "fungi")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall((text or "").lower())


@dataclass
class SearchResult:
    chunk: SourceChunk
    score: float
    rank: int


class ChunkRepository:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = ensure_dir(index_dir)
        self.chunks_path = self.index_dir / "chunks.jsonl"

    def load_chunks(self) -> list[SourceChunk]:
        return [SourceChunk.model_validate(row) for row in read_jsonl(self.chunks_path)]

    def save_chunks(self, chunks: Iterable[SourceChunk]) -> int:
        existing = {chunk.chunk_id: chunk for chunk in self.load_chunks()}
        for chunk in chunks:
            existing[chunk.chunk_id] = chunk
        rows = [chunk.model_dump(mode="json") for chunk in existing.values()]
        self.chunks_path.write_text("", encoding="utf-8")
        append_jsonl(self.chunks_path, rows)
        return len(rows)


class BM25Retriever:
    def __init__(self, chunks: list[SourceChunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(chunk.text) for chunk in chunks]
        self.doc_lens = [len(tokens) or 1 for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(len(self.doc_lens), 1)
        self.doc_freq: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            self.doc_freq.update(set(tokens))

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        query_terms = tokenize(query)
        if not query_terms:
            return []
        results: list[SearchResult] = []
        total_docs = max(len(self.chunks), 1)
        for idx, tokens in enumerate(self.doc_tokens):
            counts = Counter(tokens)
            score = 0.0
            for term in query_terms:
                df = self.doc_freq.get(term, 0)
                if df == 0:
                    continue
                idf = math.log(1 + ((total_docs - df + 0.5) / (df + 0.5)))
                tf = counts[term]
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_lens[idx] / self.avgdl)
                score += idf * ((tf * (self.k1 + 1)) / denom) if denom else 0.0
            if score > 0:
                results.append(SearchResult(self.chunks[idx], score, 0))
        results.sort(key=lambda item: item.score, reverse=True)
        ranked_results = []
        for rank, result in enumerate(results[:top_k], start=1):
            ranked_results.append(SearchResult(result.chunk, result.score, rank))
        return ranked_results


class LocalVectorIndex:
    def __init__(self, chunks: list[SourceChunk], embeddings: EmbeddingBackend) -> None:
        self.chunks = chunks
        self.embeddings = embeddings
        texts = [chunk.text for chunk in chunks]
        self.matrix = embeddings.encode(texts) if texts else np.zeros((0, 1), dtype="float32")
        self.matrix = self._normalize(self.matrix)

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        if not self.chunks:
            return []
        query_vector = self._normalize(self.embeddings.encode([query]))[0]
        scores = self.matrix @ query_vector
        order = np.argsort(scores)[::-1][:top_k]
        results: list[SearchResult] = []
        for rank, idx in enumerate(order, start=1):
            score = float(scores[idx])
            if score <= 0:
                continue
            results.append(SearchResult(self.chunks[int(idx)], score, rank))
        return results

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            return matrix
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


class ChromaVectorIndex:
    def __init__(
        self,
        chunks: list[SourceChunk],
        embeddings: EmbeddingBackend,
        persist_dir: Path,
        collection_name: str = "fungi_chunks",
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("chromadb is not installed") from exc
        self.chunks = chunks
        self.embeddings = embeddings
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(collection_name)
        if chunks:
            existing = set(self.collection.get(include=[])["ids"])
            missing = [chunk for chunk in chunks if chunk.chunk_id not in existing]
            if missing:
                vectors = embeddings.encode([chunk.text for chunk in missing]).tolist()
                self.collection.add(
                    ids=[chunk.chunk_id for chunk in missing],
                    embeddings=vectors,
                    documents=[chunk.text for chunk in missing],
                    metadatas=[self._metadata(chunk) for chunk in missing],
                )

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        if not self.chunks:
            return []
        query_vector = self.embeddings.encode([query])[0].tolist()
        response = self.collection.query(query_embeddings=[query_vector], n_results=top_k)
        by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        ids = response.get("ids", [[]])[0]
        distances = response.get("distances", [[]])[0]
        results: list[SearchResult] = []
        for rank, (chunk_id, distance) in enumerate(zip(ids, distances, strict=False), start=1):
            chunk = by_id.get(chunk_id)
            if not chunk:
                continue
            score = 1.0 / (1.0 + float(distance))
            results.append(SearchResult(chunk, score, rank))
        return results

    @staticmethod
    def _metadata(chunk: SourceChunk) -> dict[str, str | int | float | bool | None]:
        return {
            "source_id": chunk.source_id,
            "title": chunk.title,
            "url": chunk.canonical_url,
            "path": chunk.local_path,
            "section": chunk.section,
            "chunk_index": chunk.chunk_index,
        }


class HybridRetriever:
    def __init__(
        self,
        chunks: list[SourceChunk],
        *,
        embeddings: EmbeddingBackend,
        settings: Settings,
        prefer_chroma: bool = True,
        corpus_role: str | None = None,
    ) -> None:
        self.corpus_role = corpus_role
        self.chunks = filter_chunks_by_role(chunks, corpus_role)
        self.embeddings = embeddings
        self.settings = settings
        self.keyword = BM25Retriever(self.chunks)
        self.vector = self._build_vector_index(prefer_chroma)

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        prefer_chroma: bool = True,
        corpus_role: str | None = None,
    ) -> "HybridRetriever":
        settings = settings or get_settings()
        repo = ChunkRepository(settings.index_dir)
        chunks = repo.load_chunks()
        embeddings = build_embedding_backend(settings.embedding_backend, settings.embedding_model)
        return cls(
            chunks,
            embeddings=embeddings,
            settings=settings,
            prefer_chroma=prefer_chroma,
            corpus_role=corpus_role,
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        run_id: str | None = None,
    ) -> tuple[EvidencePacket, RagTrace]:
        top_k = top_k or self.settings.retrieval_top_k
        normalized = normalize_query(query)
        candidate_count = max(top_k * 4, 12)
        vector_results = self.vector.search(normalized, candidate_count)
        keyword_results = self.keyword.search(normalized, candidate_count)
        fused = self._reciprocal_rank_fusion(vector_results, keyword_results)
        selected, rejected = self._select_diverse(fused, top_k)
        items: list[EvidenceItem] = []
        for index, result in enumerate(selected):
            item = EvidenceItem(
                citation_id=index + 1,
                chunk_id=result.chunk.chunk_id,
                source_id=result.chunk.source_id,
                title=result.chunk.title,
                snippet=self._snippet(result.chunk.text),
                url=result.chunk.canonical_url,
                path=result.chunk.local_path,
                vector_score=result.score_components.get("vector", 0.0),
                keyword_score=result.score_components.get("keyword", 0.0),
                fused_score=result.fused_score,
                confidence_note=self._confidence_note(result.fused_score),
                metadata={
                    "section": result.chunk.section,
                    "page": result.chunk.page,
                    **result.chunk.metadata,
                },
            )
            items.append(item)
        packet = EvidencePacket(query=query, normalized_query=normalized, items=items)
        trace = RagTrace(
            run_id=run_id or stable_id(query),
            query=query,
            normalized_query=normalized,
            vector_candidates=self._result_rows(vector_results),
            keyword_candidates=self._result_rows(keyword_results),
            fused_candidates=[result.as_trace_row() for result in fused],
            selected_evidence=[item.model_dump(mode="json") for item in items],
            rejected_duplicates=[result.as_trace_row() for result in rejected],
        )
        return packet, trace

    def save_trace(self, trace: RagTrace, output_dir: Path) -> Path:
        path = output_dir / "rag_trace.json"
        write_json(path, trace.model_dump(mode="json"))
        return path

    def _build_vector_index(self, prefer_chroma: bool) -> LocalVectorIndex | ChromaVectorIndex:
        if prefer_chroma:
            try:
                collection_name = (
                    f"fungi_{self.corpus_role}_chunks" if self.corpus_role else "fungi_chunks"
                )
                return ChromaVectorIndex(
                    self.chunks,
                    self.embeddings,
                    self.settings.chroma_dir,
                    collection_name=collection_name,
                )
            except RuntimeError:
                pass
        return LocalVectorIndex(self.chunks, self.embeddings)

    def _result_rows(self, results: list[SearchResult]) -> list[dict[str, object]]:
        rows = []
        for result in results:
            rows.append(
                {
                "chunk_id": result.chunk.chunk_id,
                "source_id": result.chunk.source_id,
                "title": result.chunk.title,
                "score": result.score,
                "rank": result.rank,
                }
            )
        return rows

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[SearchResult],
        keyword_results: list[SearchResult],
        rrf_k: int = 60,
    ) -> list["FusedResult"]:
        fused: dict[str, FusedResult] = {}
        for channel, results in [("vector", vector_results), ("keyword", keyword_results)]:
            for result in results:
                chunk_id = result.chunk.chunk_id
                if chunk_id not in fused:
                    fused[chunk_id] = FusedResult(result.chunk)
                current = fused[chunk_id]
                current.score_components[channel] = result.score
                current.fused_score += 1.0 / (rrf_k + result.rank)
        ranked_results = sorted(fused.values(), key=lambda item: item.fused_score, reverse=True)
        for rank, result in enumerate(ranked_results, start=1):
            result.rank = rank
        return ranked_results

    def _select_diverse(
        self,
        results: list["FusedResult"],
        top_k: int,
    ) -> tuple[list["FusedResult"], list["FusedResult"]]:
        selected: list[FusedResult] = []
        rejected: list[FusedResult] = []
        source_counts: defaultdict[str, int] = defaultdict(int)
        for result in results:
            if len(selected) >= top_k:
                break
            source_already_used = (
                source_counts[result.chunk.source_id] >= 2
                and len(selected) < top_k - 1
            )
            duplicate_text = any(
                self._jaccard(result.chunk.text, prior.chunk.text) > 0.82
                for prior in selected
            )
            if source_already_used or duplicate_text:
                result.rejection_reason = (
                    "source diversity" if source_already_used else "near duplicate"
                )
                rejected.append(result)
                continue
            selected.append(result)
            source_counts[result.chunk.source_id] += 1
        if len(selected) < top_k:
            for result in results:
                if result in selected or result in rejected:
                    continue
                selected.append(result)
                if len(selected) >= top_k:
                    break
        return selected, rejected

    @staticmethod
    def _jaccard(left: str, right: str) -> float:
        a, b = set(tokenize(left)), set(tokenize(right))
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    @staticmethod
    def _snippet(text: str, max_chars: int = 620) -> str:
        clean = normalize_whitespace(text)
        return clean if len(clean) <= max_chars else clean[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _confidence_note(score: float) -> str:
        if score >= 0.03:
            return "High hybrid agreement between dense and keyword retrieval."
        if score >= 0.016:
            return "Moderate retrieval support from the indexed corpus."
        return "Low retrieval margin; use cautiously and cite precisely."


@dataclass
class FusedResult:
    chunk: SourceChunk
    fused_score: float = 0.0
    rank: int = 0
    score_components: dict[str, float] = None  # type: ignore[assignment]
    rejection_reason: str = ""

    def __post_init__(self) -> None:
        if self.score_components is None:
            self.score_components = {}

    def as_trace_row(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "chunk_id": self.chunk.chunk_id,
            "source_id": self.chunk.source_id,
            "title": self.chunk.title,
            "fused_score": self.fused_score,
            "vector_score": self.score_components.get("vector", 0.0),
            "keyword_score": self.score_components.get("keyword", 0.0),
            "rejection_reason": self.rejection_reason,
        }


def filter_chunks_by_role(chunks: list[SourceChunk], corpus_role: str | None) -> list[SourceChunk]:
    if corpus_role is None:
        return chunks
    return [
        chunk
        for chunk in chunks
        if str(chunk.metadata.get("corpus_role") or "background").lower() == corpus_role
    ]
