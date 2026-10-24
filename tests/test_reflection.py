from r1_verifiers.reflection import ReflectionVerifier


def test_reflection_verifier_detected():
    verifier = ReflectionVerifier()
    thought = (
        "Let me calculate the roots. The roots are 2 and 3. "
        "Wait, let me rethink that. If x=2, then x^2 - 5x + 6 = 4 - 10 + 6 = 0. "
        "Hold on, let me verify whether the negative root works. "
        "Checking the solution by plugging x back in: it holds."
    )
    res = verifier.verify(thought)
    assert res.reflection_count >= 2
    assert res.has_verification_step
    assert res.score >= 0.8
    assert res.thought_token_estimate > 10


def test_reflection_verifier_empty():
    verifier = ReflectionVerifier()
    res = verifier.verify("")
    assert res.reflection_count == 0
    assert not res.has_verification_step
    assert res.score == 0.0


def test_reflection_verifier_direct_no_reflection():
    verifier = ReflectionVerifier()
    thought = "Simple calculation: 2 + 2 = 4."
    res = verifier.verify(thought)
    assert res.reflection_count == 0
    assert not res.has_verification_step
    assert res.score == 0.2
