"""r1-verifiers: Lightweight reasoning verifiers & GRPO-LEAD length-scaled rewards."""

from r1_verifiers.format import FormatResult, FormatVerifier, extract_boxed_content
from r1_verifiers.grpo import (
    GRPOGroupResult,
    GRPORolloutScore,
    LengthLeadReward,
    compute_grpo_advantages,
)
from r1_verifiers.math import MathResult, MathVerifier, normalize_latex_str
from r1_verifiers.pipeline import (
    ReasoningVerificationPipeline,
    VerificationReport,
    verify_reasoning,
)
from r1_verifiers.reflection import ReflectionResult, ReflectionVerifier

__version__ = "0.1.0"

__all__ = [
    # Format
    "FormatVerifier",
    "FormatResult",
    "extract_boxed_content",
    # Math
    "MathVerifier",
    "MathResult",
    "normalize_latex_str",
    # Reflection
    "ReflectionVerifier",
    "ReflectionResult",
    # GRPO
    "LengthLeadReward",
    "GRPOGroupResult",
    "GRPORolloutScore",
    "compute_grpo_advantages",
    # Pipeline
    "ReasoningVerificationPipeline",
    "VerificationReport",
    "verify_reasoning",
]
