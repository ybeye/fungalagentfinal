from __future__ import annotations

import re
from dataclasses import dataclass


REFUSAL = (
    "I can't decide if a mushroom is safe to eat, identify a wild mushroom for real use, "
    "or give medical or dosage advice. I can still explain fungi traits, ecology, "
    "toxicity, and risks from sources. For real identification or eating decisions, ask "
    "a qualified local expert. If a person or pet ate a wild mushroom, contact poison "
    "control, emergency services, or a veterinarian right away."
)


@dataclass
class SafetyDecision:
    allowed: bool
    reason: str = ""
    refusal: str = REFUSAL


class SafetyPolicy:
    unsafe_patterns = [
        (r"\bsafe to eat\b", "eating safety"),
        (r"\b(?:okay|ok|alright) to eat\b", "eating safety"),
        (r"\bcan (?:i|we) eat\b", "eating safety"),
        (r"\bshould (?:i|we) eat\b", "eating safety"),
        (r"\bcan (?:i|we) cook\b", "eating safety"),
        (r"\b(?:cook|serve|prepare) (?:it|this|these|them).*(?:dinner|tonight|eat)\b", "eating safety"),
        (r"\b(?:eat|eating|for dinner)\b.*\b(?:mushroom|mushrooms|chanterelle|chanterelles)\b", "eating safety"),
        (r"\b(?:mushroom|mushrooms|chanterelle|chanterelles)\b.*\b(?:eat|eating|dinner)\b", "eating safety"),
        (r"\bis this edible\b", "eating safety"),
        (r"\bedible\??\b", "eating safety"),
        (r"\b(?:edible|poisonous|toxic) or (?:edible|poisonous|toxic)\b", "toxicity decision"),
        (r"\bdosage\b", "dosage advice"),
        (r"\bdose\b", "dosage advice"),
        (r"\bhow much .* take\b", "dosage advice"),
        (r"\btreat .* with\b", "medical advice"),
        (r"\bmedical advice\b", "medical advice"),
        (r"\bidentify this\b", "field ID"),
        (r"\bfield identification\b", "field ID"),
        (r"\bwhat (?:kind|species|type) of mushroom\b", "field ID"),
        (r"\bwhat mushroom is this\b", "field ID"),
        (r"\b(?:tell|identify).*(?:mushroom|fungus).*(?:picture|photo|image)\b", "field ID"),
        (r"\bconfirm .* before .* eat\b", "eating safety"),
        (r"\bpoisonous or safe\b", "toxicity decision"),
        (r"\b(?:child|kid|brother|sister|dog|cat|pet|puppy)\b.*\b(?:ate|eaten|chewed|swallowed|bit)\b.*\bmushroom", "possible poisoning"),
        (r"\bmushroom\b.*\b(?:ate|eaten|chewed|swallowed|bit)\b", "possible poisoning"),
        (r"\b(?:mushroom|mushrooms)\b.*\b(?:dangerous|poisonous|toxic)\b", "toxicity decision"),
    ]

    def __init__(self, mode: str = "strict") -> None:
        self.mode = mode
        self.compiled_patterns = []
        for pattern, reason in self.unsafe_patterns:
            compiled = re.compile(pattern, re.IGNORECASE)
            self.compiled_patterns.append((compiled, reason))

    def check(self, text: str) -> SafetyDecision:
        for pattern, reason in self.compiled_patterns:
            if pattern.search(text or ""):
                return SafetyDecision(False, f"Matched safety boundary: {reason}")
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
