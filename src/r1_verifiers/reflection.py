"""Reflection and reasoning depth verifier for thinking chains.

Detects internal self-correction, backtracking, and double-checking signals
characteristic of advanced reasoning models (DeepSeek-R1, OpenAI o1/o3).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ReflectionResult:
    """Result of analyzing reflection and self-correction within a thinking chain."""

    reflection_count: int
    has_verification_step: bool
    score: float
    detected_markers: list[str] = field(default_factory=list)
    thought_token_estimate: int = 0


DEFAULT_REFLECTION_PATTERNS = [
    r"\bwait,\s+(?:let me|is that|that's|this is|wait)\b",
    r"\bhold on\b",
    r"\blet me (?:check|verify|re-read|rethink|recalculate)\b",
    r"\balternatively\b",
    r"\bon second thought\b",
    r"\bthat (?:can't be right|doesn't make sense|is incorrect)\b",
    r"\blet's verify\b",
    r"\bdouble-check\b",
    r"\blooking back\b",
    r"\bbut wait\b",
    r"\bi made a mistake\b",
]

DEFAULT_VERIFICATION_PATTERNS = [
    r"\blet's verify whether\b",
    r"\bto check (?:my|this|the) answer\b",
    r"\bchecking (?:the result|the solution)\b",
    r"\bplugging (?:it|this|x) back in\b",
    r"\bconfirming that\b",
]


class ReflectionVerifier:
    """Quantifies self-reflection, backtracking, and verification inside thinking chains."""

    def __init__(
        self,
        reflection_patterns: list[str] | None = None,
        verification_patterns: list[str] | None = None,
    ) -> None:
        patterns = reflection_patterns or DEFAULT_REFLECTION_PATTERNS
        self.reflection_regexes = [re.compile(p, re.IGNORECASE) for p in patterns]

        v_patterns = verification_patterns or DEFAULT_VERIFICATION_PATTERNS
        self.verification_regexes = [re.compile(p, re.IGNORECASE) for p in v_patterns]

    def verify(self, thought: str) -> ReflectionResult:
        """Analyze a thinking chain for self-correction and reflection.

        Args:
            thought: The raw text inside `<think>...</think>`.

        Returns:
            ReflectionResult with count of reflections, presence of verification, and score.
        """
        if not thought or not thought.strip():
            return ReflectionResult(
                reflection_count=0,
                has_verification_step=False,
                score=0.0,
                detected_markers=[],
                thought_token_estimate=0,
            )

        text = thought.strip()
        # Rough token estimate (approx 4 chars per token)
        token_estimate = max(1, len(text.split()))

        detected: list[str] = []
        for rx in self.reflection_regexes:
            for match in rx.finditer(text):
                detected.append(match.group(0).lower().strip())

        has_verification = False
        for rx in self.verification_regexes:
            if rx.search(text):
                has_verification = True
                break

        # Compute reflection score:
        # Having at least 1-3 genuine self-reflections demonstrates non-trivial search/verification.
        count = len(detected)
        if count == 0:
            base_score = 0.2
        elif count == 1:
            base_score = 0.6
        elif count in (2, 3, 4):
            base_score = 1.0
        else:
            # Diminishing returns if model is looping excessively
            base_score = 0.8

        if has_verification:
            base_score = min(1.0, base_score + 0.2)

        return ReflectionResult(
            reflection_count=count,
            has_verification_step=has_verification,
            score=round(base_score, 3),
            detected_markers=detected,
            thought_token_estimate=token_estimate,
        )
