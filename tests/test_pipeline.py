from r1_verifiers.pipeline import ReasoningVerificationPipeline, verify_reasoning


def test_pipeline_verify_correct():
    comp = (
        "<think>Let me see: the question asks for 15 * 4. "
        "15 * 4 = 60. Wait, let me double-check. 15 * 2 = 30, 30 * 2 = 60. "
        "Checking the solution: 60.</think>\n"
        "The final answer is \\boxed{60}."
    )
    rep = verify_reasoning(comp, "60")
    assert rep.is_correct
    assert rep.composite_score > 0.8
    assert rep.format_result.is_valid
    assert rep.math_result.is_correct
    assert rep.reflection_result.reflection_count >= 1
    assert rep.thought_tokens > 0


def test_pipeline_verify_incorrect():
    comp = (
        "<think>15 * 4 is probably 50.</think>\n"
        "The answer is \\boxed{50}."
    )
    pipeline = ReasoningVerificationPipeline()
    rep = pipeline.verify(comp, "60")
    assert not rep.is_correct
    assert rep.math_result.score == 0.0
