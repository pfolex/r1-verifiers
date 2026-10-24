from r1_verifiers.format import FormatVerifier, extract_boxed_content


def test_extract_boxed_content_simple():
    text = "The final answer is \\boxed{42}."
    assert extract_boxed_content(text) == "42"


def test_extract_boxed_content_nested_braces():
    text = "We simplify to \\boxed{\\frac{3}{4}}."
    assert extract_boxed_content(text) == "\\frac{3}{4}"

    deep = "Result: \\boxed{\\sqrt{\\frac{x^{2}}{y}}}"
    assert extract_boxed_content(deep) == "\\sqrt{\\frac{x^{2}}{y}}"


def test_extract_boxed_content_none():
    assert extract_boxed_content("There is no boxed answer here.") is None


def test_format_verifier_valid():
    verifier = FormatVerifier()
    comp = "<think>Let me calculate 2+2=4.</think>\nTherefore the answer is \\boxed{4}."
    res = verifier.verify(comp)
    assert res.is_valid
    assert res.score == 1.0
    assert res.thought == "Let me calculate 2+2=4."
    assert res.extracted_answer == "4"
    assert len(res.issues) == 0


def test_format_verifier_missing_think():
    verifier = FormatVerifier(require_think=True)
    comp = "Just the answer is \\boxed{10}."
    res = verifier.verify(comp)
    assert not res.is_valid
    assert any("Missing <think>" in issue for issue in res.issues)
    assert res.extracted_answer == "10"


def test_format_verifier_unclosed_think():
    verifier = FormatVerifier()
    comp = "<think>Rambling forever without closing... \\boxed{5}"
    res = verifier.verify(comp)
    assert not res.is_valid
    assert any("Missing </think>" in issue for issue in res.issues)
