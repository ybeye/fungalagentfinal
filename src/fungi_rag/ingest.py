from __future__ import annotations

import argparse
import json
import mimetypes
import re
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup

from fungi_rag.config import Settings, get_settings
from fungi_rag.models import SourceChunk, SourceDocument
from fungi_rag.retrieval import ChunkRepository
from fungi_rag.utils import normalize_whitespace, portable_path, sha256_text, stable_id, utc_now_iso


SUPPORTED_EXTENSIONS = {".html", ".htm", ".md", ".markdown", ".txt", ".pdf"}


class DocumentIngestor:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.repo = ChunkRepository(self.settings.index_dir)

    def ingest_path(self, path: Path | str, corpus_role: str | None = None) -> list[SourceChunk]:
        target = Path(path)
        files = self._iter_files(target)
        chunks: list[SourceChunk] = []
        for file_path in files:
            chunks.extend(self.ingest_file(file_path, corpus_role=corpus_role))
        self.repo.save_chunks(chunks)
        return chunks

    def ingest_file(self, path: Path | str, corpus_role: str | None = None) -> list[SourceChunk]:
        file_path = Path(path)
        text = extract_text(file_path)
        if not text:
            return []
        metadata = read_sidecar_metadata(file_path)
        display_path = portable_path(file_path)
        role = infer_corpus_role(metadata, file_path, explicit_role=corpus_role)
        source = SourceDocument(
            source_id=metadata.get("id") or stable_id(display_path, sha256_text(text), length=12),
            title=metadata.get("title") or file_path.stem.replace("-", " ").title(),
            canonical_url=metadata.get("url"),
            local_path=display_path,
            source_type=metadata.get("source_type") or infer_source_type(file_path),
            license_note=metadata.get("license_note") or "Local/user-provided source",
            content_hash=sha256_text(text),
            retrieved_at=metadata.get("retrieved_at") or utc_now_iso(),
            metadata={
                "mime_type": mimetypes.guess_type(file_path.name)[0],
                "topics": metadata.get("topics", []),
                "corpus_role": role,
            },
        )
        pieces = chunk_text(
            text,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        chunks = []
        for index, piece in enumerate(pieces):
            chunk = SourceChunk.from_text(
                source=source,
                text=piece.text,
                chunk_index=index,
                section=piece.section,
                metadata={
                    "source_hash": source.content_hash,
                    "corpus_role": source.metadata.get("corpus_role", "background"),
                },
            )
            chunks.append(chunk)
        return chunks

    def _iter_files(self, path: Path) -> list[Path]:
        if path.is_file():
            return [path] if path.suffix.lower() in SUPPORTED_EXTENSIONS else []
        files = []
        for file_path in sorted(path.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(file_path)
        return files


class TextPiece:
    def __init__(self, text: str, section: str | None = None) -> None:
        self.text = text
        self.section = section


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".html", ".htm"}:
        return extract_html(path.read_text(encoding="utf-8", errors="ignore"))
    if suffix in {".md", ".markdown", ".txt"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        return extract_pdf(path)
    return ""


def extract_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    parts: list[str] = []
    for tag in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        text = normalize_whitespace(tag.get_text(" ", strip=True))
        if not text:
            continue
        if tag.name in {"h1", "h2", "h3"}:
            parts.append(f"\n## {text}\n")
        else:
            parts.append(text)
    return "\n\n".join(parts)


def extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return path.read_bytes().decode("utf-8", errors="ignore")
    reader = PdfReader(str(path))
    pages: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"\n## Page {index}\n{text}")
    return "\n\n".join(pages)


def chunk_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[TextPiece]:
    sections = split_sections(text)
    pieces: list[TextPiece] = []
    for section, body in sections:
        paragraphs = []
        for item in re.split(r"\n\s*\n", body):
            if item.strip():
                paragraphs.append(normalize_whitespace(item))
        buffer = ""
        for paragraph in paragraphs:
            next_buffer = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
            if len(next_buffer) <= chunk_size:
                buffer = next_buffer
                continue
            if buffer:
                pieces.append(TextPiece(buffer, section))
            buffer = paragraph
            while len(buffer) > chunk_size:
                piece = buffer[:chunk_size].rsplit(" ", 1)[0]
                pieces.append(TextPiece(piece, section))
                overlap = buffer[max(0, len(piece) - chunk_overlap) : len(piece)]
                buffer = f"{overlap} {buffer[len(piece):]}".strip()
        if buffer:
            pieces.append(TextPiece(buffer, section))
    long_pieces = []
    for piece in pieces:
        if len(piece.text) >= 80:
            long_pieces.append(piece)
    if long_pieces:
        return long_pieces
    return [TextPiece(normalize_whitespace(text))]


def split_sections(text: str) -> list[tuple[str | None, str]]:
    matches = list(re.finditer(r"(?m)^#{1,3}\s+(.+)$", text))
    if not matches:
        return [(None, text)]
    sections: list[tuple[str | None, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append((normalize_whitespace(match.group(1)), text[start:end]))
    return sections


def read_sidecar_metadata(path: Path) -> dict[str, object]:
    sidecar = path.with_suffix(path.suffix + ".metadata.json")
    if not sidecar.exists():
        return {}
    return json.loads(sidecar.read_text(encoding="utf-8"))


def infer_source_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".html", ".htm"}:
        return "html"
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".md", ".markdown"}:
        return "markdown"
    return "text"


def infer_corpus_role(
    metadata: dict[str, object],
    path: Path,
    explicit_role: str | None = None,
) -> str:
    requested_role = str(explicit_role or "").lower()
    if requested_role in {"background", "reference"}:
        return requested_role
    explicit_role = str(metadata.get("corpus_role") or "").lower()
    if explicit_role in {"background", "reference"}:
        return explicit_role
    path_parts = set()
    for part in path.parts:
        path_parts.add(part.lower())
    if "background" in path_parts:
        return "background"
    if "references" in path_parts or "reference" in path_parts:
        return "reference"
    source_id = str(metadata.get("id") or path.stem).lower()
    title = str(metadata.get("title") or path.stem).lower()
    if source_id.startswith("pmc_") or "pmc" in source_id or "journal" in title:
        return "reference"
    return "background"


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ingest local fungi source documents.")
    parser.add_argument("path", help="File or directory to ingest.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    chunks = DocumentIngestor().ingest_path(Path(args.path))
    print(f"Ingested {len(chunks)} chunks from {args.path}")


if __name__ == "__main__":
    main()
