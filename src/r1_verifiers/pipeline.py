"""End-to-end verification pipeline combining format, math, and reflection checks."""

from __future__ import annotations

from dataclasses import dataclass

from r1_verifiers.format import FormatResult, FormatVerifier
from r1_verifiers.math import MathResult, MathVerifier
from r1_verifiers.reflection import ReflectionResult, ReflectionVerifier


@dataclass
class VerificationReport:
    """Consolidated verification report for a single reasoning rollout."""

    is_correct: bool
    composite_score: float
    format_result: FormatResult
    math_result: MathResult
    reflection_result: ReflectionResult
    thought_tokens: int
    solution_tokens: int


class ReasoningVerificationPipeline:
    """Unified pipeline executing format, math, and reflection verification."""

    def __init__(
        self,
        format_verifier: FormatVerifier | None = None,
        math_verifier: MathVerifier | None = None,
        reflection_verifier: ReflectionVerifier | None = None,
    ) -> None:
        self.format_verifier = format_verifier or FormatVerifier()
        self.math_verifier = math_verifier or MathVerifier()
        self.reflection_verifier = reflection_verifier or ReflectionVerifier()

    def verify(self, completion: str, gold_answer: str) -> VerificationReport:
        """Run complete verification suite on a completion."""
        # 1. Format verification
        fmt = self.format_verifier.verify(completion)

        # 2. Mathematical verification
        mth = self.math_verifier.verify(fmt.extracted_answer, gold_answer)

        # 3. Reflection verification
        refl = self.reflection_verifier.verify(fmt.thought)

        # Token estimates
        thought_tokens = len(fmt.thought.split()) if fmt.thought else 0
        solution_tokens = len(fmt.solution.split()) if fmt.solution else 0

        # Composite score: Math correctness is primary (70%), format (20%), reflection (10%)
        composite = (
            0.70 * mth.score
            + 0.20 * fmt.score
            + 0.10 * refl.score
        )

        return VerificationReport(
            is_correct=mth.is_correct,
            composite_score=round(composite, 4),
            format_result=fmt,
            math_result=mth,
            reflection_result=refl,
            thought_tokens=thought_tokens,
            solution_tokens=solution_tokens,
        )


def verify_reasoning(completion: str, gold_answer: str) -> VerificationReport:
    """Convenience function to quickly verify a reasoning rollout."""
    pipeline = ReasoningVerificationPipeline()
    return pipeline.verify(completion, gold_answer)
