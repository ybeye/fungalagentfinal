from __future__ import annotations

import re
from dataclasses import dataclass


REFUSAL = (
    "I can't tell you whether a fungus is safe to eat, identify a wild mushroom for "
    "real-world use, or recommend medical or dosage decisions. I can help explain "
    "academic traits, ecology, toxicology, and risks from sources, but real-world "
    "identification and safety decisions should go to a qualified local expert. If a "
    "person or pet may have eaten a wild mushroom, contact poison control, emergency "
    "services, or a veterinarian right away."
)


@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    reason: str = ""
    refusal: str = REFUSAL


class SafetyPolicy:
    unsafe_patterns = [
        r"\bsafe to eat\b",
        r"\b(?:okay|ok|alright) to eat\b",
        r"\bcan (?:i|we) eat\b",
        r"\bshould (?:i|we) eat\b",
        r"\bcan (?:i|we) cook\b",
        r"\b(?:cook|serve|prepare) (?:it|this|these|them).*(?:dinner|tonight|eat)\b",
        r"\b(?:eat|eating|for dinner)\b.*\b(?:mushroom|mushrooms|chanterelle|chanterelles)\b",
        r"\b(?:mushroom|mushrooms|chanterelle|chanterelles)\b.*\b(?:eat|eating|dinner)\b",
        r"\bis this edible\b",
        r"\bedible\??\b",
        r"\b(?:edible|poisonous|toxic) or (?:edible|poisonous|toxic)\b",
        r"\bdosage\b",
        r"\bdose\b",
        r"\bhow much .* take\b",
        r"\btreat .* with\b",
        r"\bmedical advice\b",
        r"\bidentify this\b",
        r"\bfield identification\b",
        r"\bwhat (?:kind|species|type) of mushroom\b",
        r"\bwhat mushroom is this\b",
        r"\b(?:tell|identify).*(?:mushroom|fungus).*(?:picture|photo|image)\b",
        r"\bconfirm .* before .* eat\b",
        r"\bpoisonous or safe\b",
        r"\b(?:child|kid|brother|sister|dog|cat|pet|puppy)\b.*\b(?:ate|eaten|chewed|swallowed|bit)\b.*\bmushroom",
        r"\bmushroom\b.*\b(?:ate|eaten|chewed|swallowed|bit)\b",
        r"\b(?:mushroom|mushrooms)\b.*\b(?:dangerous|poisonous|toxic)\b",
    ]

    def __init__(self, mode: str = "strict") -> None:
        self.mode = mode
        self._compiled = [re.compile(pattern, re.IGNORECASE) for pattern in self.unsafe_patterns]

    def check(self, text: str) -> SafetyDecision:
        for pattern in self._compiled:
            if pattern.search(text or ""):
                return SafetyDecision(False, f"Matched safety boundary: {pattern.pattern}")
        return SafetyDecision(True)

    def validate_response(self, text: str) -> list[str]:
        errors: list[str] = []
        lowered = (text or "").lower()
        risky_claims = [
            "safe to eat",
            "you can eat",
            "recommended dose",
            "take this mushroom",
            "definitely edible",
            "field identification confirms",
        ]
        for claim in risky_claims:
            if claim in lowered:
                errors.append(f"Response contains unsafe claim: {claim!r}")
        return errors
