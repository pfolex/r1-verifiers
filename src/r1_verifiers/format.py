"""Format verifier for reasoning rollouts (e.g. DeepSeek-R1, QwQ).

Validates `<think>...</think>` tag structure and extracts boxed answers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class FormatResult:
    """Result of format verification on a reasoning rollout."""

    is_valid: bool
    score: float
    thought: str = ""
    solution: str = ""
    extracted_answer: str | None = None
    issues: list[str] = field(default_factory=list)


def extract_boxed_content(text: str) -> str | None:
    """Extract the outermost content from \\boxed{...}, handling nested braces.

    Returns the inside of the last \\boxed{...} tag found in the text, or None.
    """
    marker = "\\boxed{"
    last_idx = text.rfind(marker)
    if last_idx == -1:
        return None

    start = last_idx + len(marker)
    brace_depth = 1
    i = start
    while i < len(text) and brace_depth > 0:
        if text[i] == "{":
            brace_depth += 1
        elif text[i] == "}":
            brace_depth -= 1
        i += 1

    if brace_depth == 0:
        return text[start : i - 1].strip()
    return None


class FormatVerifier:
    """Verifies that completion strictly conforms to reasoning model format."""

    def __init__(
        self,
        require_think: bool = True,
        require_boxed: bool = True,
        max_trailing_chars: int = 500,
    ) -> None:
        self.require_think = require_think
        self.require_boxed = require_boxed
        self.max_trailing_chars = max_trailing_chars

    def verify(self, completion: str) -> FormatResult:
        """Verify the formatting of a model completion.

        Args:
            completion: Full text output from the reasoning model.

        Returns:
            FormatResult with validity status, partial score, thoughts, and extracted answer.
        """
        text = completion.strip()
        issues: list[str] = []
        score = 1.0

        thought = ""
        solution = text

        has_think_open = "<think>" in text
        has_think_close = "</think>" in text

        if self.require_think:
            if not has_think_open:
                issues.append("Missing <think> opening tag.")
                score -= 0.5
            if not has_think_close:
                issues.append("Missing </think> closing tag.")
                score -= 0.5

            if has_think_open and has_think_close:
                think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
                matches = think_pattern.findall(text)
                if len(matches) > 1:
                    issues.append("Multiple <think> blocks found.")
                    score -= 0.2

                if matches:
                    thought = matches[-1].strip()
                    if not thought:
                        issues.append("Empty thinking chain inside <think> tags.")
                        score -= 0.3
                    # Solution is everything after the final </think> tag
                    close_idx = text.rfind("</think>")
                    solution = text[close_idx + len("</think>") :].strip()

        # Extract answer
        extracted_answer = extract_boxed_content(solution)
        if not extracted_answer and extract_boxed_content(text):
            # Sometimes boxed is inside the text or improperly delimited
            extracted_answer = extract_boxed_content(text)
            if self.require_think and has_think_close:
                issues.append("Boxed answer found inside thinking block instead of final solution.")
                score -= 0.2

        if self.require_boxed and not extracted_answer:
            # Fallback regex for "Answer: X" or "The answer is X"
            alt_match = re.search(r"(?:The answer is|Answer:)\s*([^\n\.]+)", solution, re.IGNORECASE)
            if alt_match:
                extracted_answer = alt_match.group(1).strip()
                issues.append("Answer extracted from fallback pattern instead of \\boxed{}.")
                score -= 0.2
            else:
                issues.append("No \\boxed{...} or clear answer delimiter found.")
                score -= 0.5

        final_score = max(0.0, score)
        is_valid = len(issues) == 0

        return FormatResult(
            is_valid=is_valid,
            score=final_score,
            thought=thought,
            solution=solution,
            extracted_answer=extracted_answer,
            issues=issues,
        )
