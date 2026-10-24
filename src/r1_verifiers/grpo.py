"""GRPO and GRPO-LEAD length-scaled reward functions for reasoning models.

Implements length-scaled accuracy rewards to combat overthinking loops in
DeepSeek-R1 and reasoning models as formulated in GRPO-LEAD (arXiv:2504.09696).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass
class GRPORolloutScore:
    """Individual rollout reward and advantage within a group."""

    index: int
    is_correct: bool
    length: int
    length_z_score: float
    raw_reward: float
    advantage: float


@dataclass
class GRPOGroupResult:
    """Aggregated group statistics and scored rollouts."""

    mean_length: float
    std_length: float
    mean_reward: float
    std_reward: float
    scores: list[GRPORolloutScore]


class LengthLeadReward:
    """Implements GRPO-LEAD length-scaled accuracy reward and advantage calculation.

    For correct answers:
        R(y_i) = exp(-alpha * z_i)
    where z_i = (L_i - mu_L) / (sigma_L + eps).

    For incorrect answers:
        R(y_i) = incorrect_penalty (default: -1.0).
    """

    def __init__(
        self,
        alpha: float = 0.5,
        incorrect_penalty: float = -1.0,
        eps: float = 1e-6,
    ) -> None:
        self.alpha = alpha
        self.incorrect_penalty = incorrect_penalty
        self.eps = eps

    def score_group(
        self,
        correctness: Sequence[bool],
        lengths: Sequence[int],
    ) -> GRPOGroupResult:
        """Score a group of reasoning rollouts and compute GRPO advantages.

        Args:
            correctness: Boolean flags indicating correctness for each rollout.
            lengths: Token or character length of each completion/thought.

        Returns:
            GRPOGroupResult containing individual scores, z-scores, rewards, and advantages.
        """
        n = len(correctness)
        if n == 0:
            return GRPOGroupResult(0.0, 0.0, 0.0, 0.0, [])

        if len(lengths) != n:
            raise ValueError(
                f"Length mismatch: {n} correctness values vs {len(lengths)} lengths."
            )

        # 1. Compute length statistics
        mean_l = sum(lengths) / n
        var_l = sum((x - mean_l) ** 2 for x in lengths) / n
        std_l = math.sqrt(var_l)

        # 2. Compute raw rewards
        raw_rewards: list[float] = []
        z_scores: list[float] = []

        for is_corr, length in zip(correctness, lengths, strict=False):
            if std_l > self.eps:
                z = (length - mean_l) / (std_l + self.eps)
            else:
                z = 0.0
            z_scores.append(z)

            if is_corr:
                # Shorter correct completions receive exponentially higher reward
                reward = math.exp(-self.alpha * z)
            else:
                reward = self.incorrect_penalty
            raw_rewards.append(reward)

        # 3. Compute reward statistics across group
        mean_r = sum(raw_rewards) / n
        var_r = sum((r - mean_r) ** 2 for r in raw_rewards) / n
        std_r = math.sqrt(var_r)

        # 4. Compute GRPO normalized advantages: A_i = (R_i - mu_R) / (sigma_R + eps)
        scores: list[GRPORolloutScore] = []
        for i, (is_corr, length, z, r) in enumerate(
            zip(correctness, lengths, z_scores, raw_rewards, strict=False)
        ):
            if std_r > self.eps:
                adv = (r - mean_r) / (std_r + self.eps)
            else:
                adv = 0.0

            scores.append(
                GRPORolloutScore(
                    index=i,
                    is_correct=is_corr,
                    length=length,
                    length_z_score=round(z, 4),
                    raw_reward=round(r, 4),
                    advantage=round(adv, 4),
                )
            )

        return GRPOGroupResult(
            mean_length=round(mean_l, 2),
            std_length=round(std_l, 2),
            mean_reward=round(mean_r, 4),
            std_reward=round(std_r, 4),
            scores=scores,
        )


def compute_grpo_advantages(
    correctness: Sequence[bool],
    lengths: Sequence[int],
    alpha: float = 0.5,
) -> list[float]:
    """Convenience helper returning just the advantage values for an RL rollout step."""
    scorer = LengthLeadReward(alpha=alpha)
    res = scorer.score_group(correctness, lengths)
    return [s.advantage for s in res.scores]
