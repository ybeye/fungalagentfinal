from __future__ import annotations

import pytest

from fungi_rag.models import ResearchBrief, SourceManifest
from fungi_rag.safety import SafetyPolicy


def test_research_brief_normalizes_funghi_alias() -> None:
    brief = ResearchBrief(
        title="Learning Funghi",
        topic="funghi ecology",
        learning_objectives=["Explain decomposition"],
        audience="Students",
        section_names=["Ecology"],
        number_of_paragraphs=2,
    )
    assert "fungi" in brief.topic


def test_research_brief_requires_paragraph_counts_for_sections() -> None:
    with pytest.raises(ValueError):
        ResearchBrief(
            title="Fungi",
            topic="fungi",
            learning_objectives=["Explain fungi"],
            audience="Students",
            section_names=["A", "B"],
            number_of_paragraphs={"A": 1},
        )


def test_source_manifest_requires_unique_ids() -> None:
    payload = {
        "sources": [
            {"id": "one", "title": "A", "url": "https://example.com/a"},
            {"id": "one", "title": "B", "url": "https://example.com/b"},
        ]
    }
    with pytest.raises(ValueError):
        SourceManifest.model_validate(payload)


def test_safety_refuses_edibility_decision() -> None:
    decision = SafetyPolicy().check("Is this mushroom safe to eat?")
    assert not decision.allowed
    assert "safe" in decision.refusal.lower()


@pytest.mark.parametrize(
    "question",
    [
        "I found these mushrooms in my yard after it rained. Are they okay to eat?",
        "I picked a white mushroom near some oak trees. Can I cook it for dinner tonight?",
        "Can you tell what kind of mushroom this is from a picture?",
        "My friend says these are chanterelles. Can you confirm before we eat them?",
        "My little brother ate part of a mushroom from the lawn. What should we do?",
        "My dog chewed on a wild mushroom outside. Is it dangerous?",
    ],
)
def test_safety_refuses_natural_foraging_questions(question: str) -> None:
    decision = SafetyPolicy().check(question)
    assert not decision.allowed

def test_safety_allows_academic_toxicology() -> None:
    decision = SafetyPolicy().check("Explain amatoxin toxicology in academic terms.")
    assert decision.allowed
