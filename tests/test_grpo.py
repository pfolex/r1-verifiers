import math

from r1_verifiers.grpo import LengthLeadReward, compute_grpo_advantages


def test_length_lead_reward_ranking():
    """Verify that shorter correct completions get higher reward than longer correct completions."""
    scorer = LengthLeadReward(alpha=0.5, incorrect_penalty=-1.0)
    # Rollout 0: concise & correct (100 tokens)
    # Rollout 1: rambling & correct (1000 tokens)
    # Rollout 2: incorrect (500 tokens)
    correctness = [True, True, False]
    lengths = [100, 1000, 500]

    res = scorer.score_group(correctness, lengths)
    scores = res.scores

    # Verify ranking: Short correct > Long correct > Incorrect
    assert scores[0].raw_reward > scores[1].raw_reward
    assert scores[1].raw_reward > scores[2].raw_reward
    assert scores[2].raw_reward == -1.0

    # Verify advantages: highest advantage goes to short correct
    assert scores[0].advantage > scores[1].advantage
    assert scores[1].advantage > scores[2].advantage

    # Advantages across group approximately mean-zero
    mean_adv = sum(s.advantage for s in scores) / len(scores)
    assert math.isclose(mean_adv, 0.0, abs_tol=1e-3)


def test_compute_grpo_advantages_helper():
    advs = compute_grpo_advantages([True, False], [200, 200], alpha=0.5)
    assert len(advs) == 2
    assert advs[0] > 0
    assert advs[1] < 0


def test_zero_variance_lengths():
    """Handle identical lengths without ZeroDivisionError."""
    scorer = LengthLeadReward()
    res = scorer.score_group([True, True], [300, 300])
    assert len(res.scores) == 2
    assert res.scores[0].length_z_score == 0.0
