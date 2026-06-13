from __future__ import annotations

from pathlib import Path

from fungi_rag.config import Settings
from fungi_rag.embeddings import HashingEmbeddingBackend
from fungi_rag.ingest import DocumentIngestor, infer_corpus_role
from fungi_rag.models import SourceChunk, SourceManifestEntry
from fungi_rag.retrieval import ChunkRepository, HybridRetriever, filter_chunks_by_role
from fungi_rag.sources import SourceDownloader, corpus_role_for


def settings_for(tmp_path: Path) -> Settings:
    settings = Settings(
        embedding_backend="hashing",
        chroma_dir=tmp_path / "chroma",
        upload_dir=tmp_path / "uploads",
        background_dir=tmp_path / "background",
        references_dir=tmp_path / "references",
        source_raw_dir=tmp_path / "sources",
        source_state_path=tmp_path / "sources.jsonl",
        index_dir=tmp_path / "index",
        output_dir=tmp_path / "outputs",
        chunk_size=300,
        chunk_overlap=40,
    )
    settings.ensure_directories()
    return settings


def test_ingest_chunks_and_hybrid_retrieval(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    source = tmp_path / "data" / "fungi.txt"
    source.parent.mkdir()
    source.write_text(
        """
        # Fungal morphology

        Hyphae form threadlike filaments. A mycelium expands through a substrate and
        helps fungi absorb nutrients by increasing surface area.

        # Decomposition

        Saprotrophic fungi decompose lignin and cellulose, returning nutrients to soil.
        """,
        encoding="utf-8",
    )
    settings = settings_for(tmp_path)
    chunks = DocumentIngestor(settings).ingest_path(source)
    assert chunks

    stored = ChunkRepository(settings.index_dir).load_chunks()
    assert stored[0].local_path
    assert not Path(stored[0].local_path).is_absolute()
    retriever = HybridRetriever(
        stored,
        embeddings=HashingEmbeddingBackend(),
        settings=settings,
        prefer_chroma=False,
    )
    packet, trace = retriever.retrieve("How do hyphae help nutrient absorption?", top_k=2)
    assert packet.items
    assert "hyphae" in packet.items[0].snippet.lower()
    assert trace.vector_candidates or trace.keyword_candidates


def test_retriever_normalizes_funghi_query(tmp_path: Path) -> None:
    source = tmp_path / "ecology.txt"
    source.write_text("Fungi are important decomposers in forest nutrient cycles.", encoding="utf-8")
    settings = settings_for(tmp_path)
    DocumentIngestor(settings).ingest_path(source)
    chunks = ChunkRepository(settings.index_dir).load_chunks()
    retriever = HybridRetriever(
        chunks,
        embeddings=HashingEmbeddingBackend(),
        settings=settings,
        prefer_chroma=False,
    )
    packet, _trace = retriever.retrieve("funghi decomposers", top_k=1)
    assert packet.normalized_query == "fungi decomposers"
    assert packet.items


def test_sources_split_into_background_and_reference_roles() -> None:
    assert infer_corpus_role({"id": "openstax_bio_24_1"}, Path("openstax.html")) == "background"
    assert infer_corpus_role({"id": "pmc_fungal_traits"}, Path("pmc.html")) == "reference"
    assert infer_corpus_role({}, Path("data/background/book.txt")) == "background"
    assert infer_corpus_role({}, Path("data/references/paper.txt")) == "reference"
    assert infer_corpus_role({}, Path("anything.txt"), explicit_role="reference") == "reference"


def test_source_downloader_copies_seed_files_to_project_corpus_folders(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    entry = SourceManifestEntry(
        id="pmc_test",
        title="Reference Paper",
        url="https://example.com/paper",
        corpus_role="reference",
    )
    source_path = settings.source_raw_dir / "pmc-test.html"
    sidecar_path = source_path.with_suffix(source_path.suffix + ".metadata.json")
    source_path.write_text("<h1>Reference Paper</h1>", encoding="utf-8")
    sidecar_path.write_text('{"id": "pmc_test", "title": "Reference Paper"}', encoding="utf-8")

    downloader = SourceDownloader(settings, rate_limit_seconds=0)
    copied_path = downloader.copy_entry_to_project_folder(
        source_path,
        sidecar_path,
        entry,
    )

    assert copied_path == settings.references_dir / "pmc-test.html"
    assert copied_path.exists()
    assert '"corpus_role": "reference"' in copied_path.with_suffix(".html.metadata.json").read_text(
        encoding="utf-8"
    )
    background_entry = SourceManifestEntry(id="openstax", title="Book", url="https://x.test")
    assert corpus_role_for(background_entry) == "background"


def test_filter_chunks_by_corpus_role() -> None:
    chunks = [
        SourceChunk(
            chunk_id="background:0",
            source_id="background",
            title="Background source",
            text="General fungi background.",
            chunk_index=0,
            content_hash="hash1",
            metadata={"corpus_role": "background"},
        ),
        SourceChunk(
            chunk_id="reference:0",
            source_id="reference",
            title="Reference paper",
            text="Academic fungi reference.",
            chunk_index=0,
            content_hash="hash2",
            metadata={"corpus_role": "reference"},
        ),
    ]
    background_chunks = filter_chunks_by_role(chunks, "background")
    reference_chunks = filter_chunks_by_role(chunks, "reference")

    assert [chunk.source_id for chunk in background_chunks] == ["background"]
    assert [chunk.source_id for chunk in reference_chunks] == ["reference"]
