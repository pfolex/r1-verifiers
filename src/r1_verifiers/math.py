"""Mathematical equivalence verifier for reasoning rollouts.

Compares extracted answers against gold reference solutions using symbolic
and numerical normalization.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from sympy import simplify, sympify


@dataclass
class MathResult:
    """Result of mathematical verification."""

    is_correct: bool
    score: float
    extracted_answer: str | None
    gold_answer: str
    error: str | None = None


def normalize_latex_str(text: str) -> str:
    """Clean common LaTeX formatting artifacts for comparison."""
    if not text:
        return ""
    s = text.strip()
    # Strip enclosing dollar signs
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    # Remove text wrappers
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", s)
    # Remove sizing commands
    s = re.sub(r"\\left|\\right", "", s)
    # Convert fractions \frac{a}{b} -> (a)/(b)
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\dfrac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", s)
    # Common math symbols
    s = s.replace(r"\cdot", "*").replace(r"\times", "*")
    s = s.replace(r"\pi", "pi")
    s = s.replace(r"\infty", "oo")
    s = s.replace(r"\%", "%")
    s = s.replace(" ", "")
    return s


class MathVerifier:
    """Verifies correctness of mathematical answers."""

    def __init__(self, tolerance: float = 1e-6) -> None:
        self.tolerance = tolerance

    def _try_numeric_eq(self, a_str: str, b_str: str) -> bool:
        """Attempt pure numeric comparison with floating point tolerance."""
        # Strip trailing percentage signs
        is_pct_a = a_str.endswith("%")
        is_pct_b = b_str.endswith("%")
        raw_a = a_str[:-1] if is_pct_a else a_str
        raw_b = b_str[:-1] if is_pct_b else b_str

        try:
            val_a = float(raw_a)
            val_b = float(raw_b)
            if is_pct_a and not is_pct_b:
                val_a /= 100.0
            elif is_pct_b and not is_pct_a:
                val_b /= 100.0
            return math.isclose(val_a, val_b, rel_tol=self.tolerance, abs_tol=self.tolerance)
        except ValueError:
            return False

    def _try_symbolic_eq(self, a_str: str, b_str: str) -> bool:
        """Attempt symbolic equivalence using SymPy."""
        # Handle equations "LHS = RHS"
        if "=" in a_str and "=" in b_str:
            parts_a = a_str.split("=", 1)
            parts_b = b_str.split("=", 1)
            expr_a = sympify(parts_a[0]) - sympify(parts_a[1])
            expr_b = sympify(parts_b[0]) - sympify(parts_b[1])
            diff = simplify(expr_a - expr_b)
            if diff == 0:
                return True
            # Check flipped
            diff_flipped = simplify(expr_a + expr_b)
            return diff_flipped == 0

        expr_a = sympify(a_str)
        expr_b = sympify(b_str)
        diff = simplify(expr_a - expr_b)
        if diff == 0:
            return True

        # Check numeric evaluation
        try:
            val = float(diff.evalf())
            return abs(val) < self.tolerance
        except (TypeError, ValueError):
            return False

    def verify(self, extracted: str | None, gold: str) -> MathResult:
        """Verify if extracted answer matches the gold reference.

        Args:
            extracted: Answer candidate extracted from model completion.
            gold: Ground-truth reference answer.

        Returns:
            MathResult indicating match status and score (1.0 for match, 0.0 otherwise).
        """
        if extracted is None or not extracted.strip():
            return MathResult(
                is_correct=False,
                score=0.0,
                extracted_answer=extracted,
                gold_answer=gold,
                error="No answer extracted",
            )

        clean_gold = normalize_latex_str(gold)
        clean_ext = normalize_latex_str(extracted)

        # 1. Exact string match
        if clean_gold == clean_ext:
            return MathResult(is_correct=True, score=1.0, extracted_answer=extracted, gold_answer=gold)

        # 2. Direct numeric comparison
        if self._try_numeric_eq(clean_ext, clean_gold):
            return MathResult(is_correct=True, score=1.0, extracted_answer=extracted, gold_answer=gold)

        # 3. Symbolic SymPy comparison
        try:
            if self._try_symbolic_eq(clean_ext, clean_gold):
                return MathResult(is_correct=True, score=1.0, extracted_answer=extracted, gold_answer=gold)
        except Exception as e:
            # Symbolic parse failure
            return MathResult(
                is_correct=False,
                score=0.0,
                extracted_answer=extracted,
                gold_answer=gold,
                error=f"Symbolic comparison error: {e}",
            )

        return MathResult(is_correct=False, score=0.0, extracted_answer=extracted, gold_answer=gold)
