from r1_verifiers.math import MathVerifier, normalize_latex_str


def test_normalize_latex_str():
    assert normalize_latex_str("$42$") == "42"
    assert normalize_latex_str("\\text{answer}") == "answer"
    assert normalize_latex_str("\\left( 5 \\right)") == "(5)"
    assert normalize_latex_str("\\frac{1}{2}") == "((1)/(2))"


def test_math_verifier_exact():
    verifier = MathVerifier()
    res = verifier.verify("42", "42")
    assert res.is_correct
    assert res.score == 1.0


def test_math_verifier_numeric_float():
    verifier = MathVerifier()
    res = verifier.verify("3.14159", "3.141590")
    assert res.is_correct

    res_pct = verifier.verify("25%", "0.25")
    assert res_pct.is_correct


def test_math_verifier_fraction_symbolic():
    verifier = MathVerifier()
    res = verifier.verify("\\frac{1}{2}", "0.5")
    assert res.is_correct

    res2 = verifier.verify("2*(x + 3)", "2*x + 6")
    assert res2.is_correct


def test_math_verifier_wrong():
    verifier = MathVerifier()
    res = verifier.verify("10", "15")
    assert not res.is_correct
    assert res.score == 0.0


def test_math_verifier_none():
    verifier = MathVerifier()
    res = verifier.verify(None, "42")
    assert not res.is_correct
    assert res.score == 0.0
