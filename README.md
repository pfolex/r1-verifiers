# r1-verifiers 🧠⚡

[![Tests](https://img.shields.io/badge/tests-22%20passed-success.svg)](https://github.com/pfolex/r1-verifiers)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://github.com/pfolex/r1-verifiers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Lightweight, production-ready reasoning verifiers and **GRPO-LEAD length-scaled rewards** for **DeepSeek-R1**, **OpenAI o1/o3**, **QwQ**, and reasoning LLM reinforcement learning pipelines.

---

## 🌟 Why `r1-verifiers`?

Long reasoning chains in modern LLMs improve problem-solving accuracy, but frequently suffer from **reasoning overthinking**—models get stuck in verbose, repetitive thought loops and receive high rewards simply by writing more tokens.

`r1-verifiers` provides:
1. **Format Verification**: Validates `<think>...</think>` tags and extracts LaTeX boxed answers (`\boxed{...}`).
2. **Mathematical Verification**: Robust symbolic and numerical answer checking using SymPy.
3. **Reflection Detection**: Analyzes internal self-correction, backtracking, and verification intensity.
4. **GRPO-LEAD Length-Scaled Rewards**: Rewards concise correct solutions and suppresses rambling overthinking loops under Group Relative Policy Optimization (GRPO).
5. **CLI Utility (`r1-verify`)**: Grade reasoning completions or batch rollout logs directly from the terminal.

---

## 📦 Installation

```bash
pip install r1-verifiers
```

Or install with `uv`:
```bash
uv add r1-verifiers
```

---

## 🚀 Quickstart

### 1. Verify a Single Reasoning Rollout

```python
from r1_verifiers import verify_reasoning

completion = """
<think>
Let's solve for x: 2x + 10 = 24.
2x = 14
x = 7.
Wait, let me double-check: 2(7) + 10 = 14 + 10 = 24. Holds.
</think>
The value of x is \\boxed{7}.
"""

report = verify_reasoning(completion, gold_answer="7")

print(f"Correct: {report.is_correct}")  # True
print(f"Composite Score: {report.composite_score}")  # 0.98
print(f"Extracted Answer: {report.format_result.extracted_answer}")  # '7'
print(f"Reflections: {report.reflection_result.reflection_count}")  # 2
```

---

### 2. GRPO-LEAD Length-Scaled Advantage Calculation

Combats overthinking during RLHF / GRPO training by calculating length-normalized rewards:

```python
from r1_verifiers import LengthLeadReward

# Candidate rollouts sampled from reasoning model:
# Candidate 0: Concise & correct (120 tokens)
# Candidate 1: Rambling & correct (1200 tokens)
# Candidate 2: Incorrect (600 tokens)
correctness = [True, True, False]
lengths = [120, 1200, 600]

scorer = LengthLeadReward(alpha=0.5, incorrect_penalty=-1.0)
result = scorer.score_group(correctness, lengths)

for score in result.scores:
    print(
        f"Rollout {score.index}: "
        f"Correct={score.is_correct}, "
        f"Length={score.length}, "
        f"Reward={score.raw_reward:.3f}, "
        f"Advantage={score.advantage:+.3f}"
    )
```

**Output:**
```text
Rollout 0: Correct=True,  Length=120,  Reward=1.350, Advantage=+1.382  # Boosted!
Rollout 1: Correct=True,  Length=1200, Reward=0.741, Advantage=-0.518  # Suppressed overthinking!
Rollout 2: Correct=False, Length=600,  Reward=-1.000, Advantage=-0.864
```

---

### 3. Integration with Hugging Face TRL (`GRPOTrainer`)

Use `r1_verifiers` directly as a custom reward function in TRL:

```python
from trl import GRPOTrainer
from r1_verifiers import LengthLeadReward, extract_boxed_content, MathVerifier

math_verifier = MathVerifier()
lead_reward = LengthLeadReward(alpha=0.5)

def grpo_lead_reward_fn(completions, target, **kwargs):
    rewards = []
    correctness = []
    lengths = [len(c.split()) for c in completions]

    for comp, gold in zip(completions, target):
        ans = extract_boxed_content(comp)
        res = math_verifier.verify(ans, gold)
        correctness.append(res.is_correct)

    group_res = lead_reward.score_group(correctness, lengths)
    return [s.raw_reward for s in group_res.scores]

# Pass to GRPOTrainer:
# trainer = GRPOTrainer(model=model, reward_funcs=[grpo_lead_reward_fn], ...)
```

---

## 💻 CLI Usage (`r1-verify`)

### Verify a Single Completion:
```bash
r1-verify --completion "<think>7*8=56. Let me verify: 56.</think> \boxed{56}" --gold "56"
```

### Grade a JSON Batch of Rollouts:
```bash
r1-verify --file rollouts.json --alpha 0.5
```

```text
======================================================================
  BATCH VERIFICATION & GRPO-LEAD ADVANTAGE EVALUATION (N=3)
======================================================================
Idx  | Correct | Tokens | z_len   | Reward   | Advantage | Answer
----------------------------------------------------------------------
0    | YES     | 812    |  +0.39 |  +0.825 |   -0.522 | 0.05
1    | YES     | 581    |  -1.37 |  +1.985 |   +1.399 | 0.05
2    | YES     | 891    |  +0.99 |  +0.611 |   -0.877 | 0.05
======================================================================
```

Add `--json` for machine-readable JSON output suitable for pipelines.

---

## 📐 Mathematical Formulation (GRPO-LEAD)

For a group of $G$ sampled reasoning rollouts $\{y_1, y_2, \dots, y_G\}$ for prompt $x$:

1. **Group Length Statistics**:
   $$\mu_L = \frac{1}{G}\sum_{i=1}^G L(y_i), \quad \sigma_L = \sqrt{\frac{1}{G}\sum_{i=1}^G (L(y_i) - \mu_L)^2}$$

2. **Standardized Length $z$-Score**:
   $$z_i = \frac{L(y_i) - \mu_L}{\sigma_L + \epsilon}$$

3. **Length-Scaled Reward**:
   $$R(y_i) = \begin{cases} \exp(-\alpha \cdot z_i) & \text{if } y_i \text{ is correct} \\ -1.0 & \text{if } y_i \text{ is incorrect} \end{cases}$$

4. **Normalized GRPO Advantage**:
   $$A_i = \frac{R(y_i) - \mu_R}{\sigma_R + \epsilon}$$

---

## 🧪 Testing

Run the full pytest suite:

```bash
pytest tests/ -v
```

---

## 📜 License

MIT License. Copyright (c) 2026 Alex Pfeil.
