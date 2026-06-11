from __future__ import annotations

import re
from dataclasses import dataclass, field

from fungi_rag.models import EvidencePacket


@dataclass
class CitationAuditResult:
    ok: bool
    cited_ids: list[int] = field(default_factory=list)
    missing_ids: list[int] = field(default_factory=list)
    unknown_ids: list[int] = field(default_factory=list)
    unsupported_sentences: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "cited_ids": self.cited_ids,
            "missing_ids": self.missing_ids,
            "unknown_ids": self.unknown_ids,
            "unsupported_sentences": self.unsupported_sentences,
        }


class CitationAuditor:
    citation_pattern = re.compile(r"\[(\d+)\]")
    parenthetical_citation_pattern = re.compile(r"(?<!\w)\((\d{1,3})\)(?!\w)")
    sentence_pattern = re.compile(r"(?<=[.!?])\s+")

    def audit(self, text: str, evidence: EvidencePacket, require_all: bool = False) -> CitationAuditResult:
        available = {item.citation_id for item in evidence.items}
        text = normalize_numeric_citations(text, available)
        cited = sorted({int(match.group(1)) for match in self.citation_pattern.finditer(text or "")})
        unknown = sorted(set(cited) - available)
        missing = sorted(available - set(cited)) if require_all else []
        unsupported = self._unsupported_sentences(text)
        ok = not unknown and not missing and not unsupported
        return CitationAuditResult(
            ok=ok,
            cited_ids=cited,
            missing_ids=missing,
            unknown_ids=unknown,
            unsupported_sentences=unsupported,
        )

    def _unsupported_sentences(self, text: str) -> list[str]:
        unsupported: list[str] = []
        for sentence in self.sentence_pattern.split(text or ""):
            clean = sentence.strip()
            if len(clean) < 80:
                continue
            if self.citation_pattern.search(clean):
                continue
            if clean.startswith(("#", "-", "*")):
                continue
            unsupported.append(clean[:220])
        return unsupported[:10]


def normalize_numeric_citations(text: str, available_ids: set[int] | None = None) -> str:
    """Convert obvious citation-only parentheticals like (5) to [5]."""
    available = available_ids or set()

    def replace(match: re.Match[str]) -> str:
        citation_id = int(match.group(1))
        if available and citation_id not in available:
            return match.group(0)
        return f"[{citation_id}]"

    return CitationAuditor.parenthetical_citation_pattern.sub(replace, text or "")


def format_references(evidence: EvidencePacket) -> str:
    lines = ["## References"]
    for item in evidence.items:
        locator = item.url or item.path or item.source_id
        lines.append(f"[{item.citation_id}] {item.title}. {locator}")
    return "\n".join(lines)
