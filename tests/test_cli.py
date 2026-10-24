import json
import tempfile

from r1_verifiers.cli import run_batch, run_single


def test_cli_run_single(capsys):
    comp = "<think>2+2=4</think>\n\\boxed{4}"
    code = run_single(comp, "4", as_json=True)
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["is_correct"]
    assert data["format"]["extracted_answer"] == "4"


def test_cli_run_batch(capsys):
    items = [
        {"completion": "<think>short</think>\\boxed{10}", "gold": "10"},
        {"completion": "<think>very long rambling thought here</think>\\boxed{10}", "gold": "10"},
        {"completion": "<think>wrong</think>\\boxed{99}", "gold": "10"},
    ]
    with tempfile.NamedTemporaryFile("w+", suffix=".json") as tmp:
        json.dump(items, tmp)
        tmp.flush()
        code = run_batch(tmp.name, alpha=0.5, as_json=True)
        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data["rollouts"]) == 3
        # Short correct gets higher reward than long correct
        assert data["rollouts"][0]["raw_reward"] > data["rollouts"][1]["raw_reward"]
        assert data["rollouts"][2]["is_correct"] is False
