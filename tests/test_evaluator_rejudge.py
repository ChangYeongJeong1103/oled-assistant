"""Offline regression checks for versioned grading and saved-result rejudging."""
import hashlib
import json
from pathlib import Path

import pytest

import evaluate_agent as ev
import evaluate_multiturn as multi
from evaluate_agent import add_scores, public_copy


@pytest.mark.parametrize("claims", [[], [{
    "claim": "FIrpic is blue.", "verdict": "supported", "reason": "Missing quote",
}]])
def test_v3_incomplete_extraction_cannot_pass_grounding_gate(claims):
    record = {
        "id": "x", "category": "normal", "pair_id": None, "answer": "FIrpic is blue.",
        "expected_mode": "RAG", "mode": "RAG", "used_files": [],
        "latency_s": 1, "cost_usd": 0, "searches": 1, "llm_calls": 1,
        "prompt_tokens": 1, "completion_tokens": 1,
    }
    scored = add_scores({}, record, {
        "claims": claims, "key_points": [], "judge_policy": ev.JUDGE_POLICY,
    })
    summary = ev.summarize([scored])
    assert not scored["grounding_valid"]
    assert summary["grounding_evaluation_complete"] is False
    assert summary["grounding_invalid_answers"] == 1
    assert summary["unsupported_claim_rate"] is None


def test_grounding_and_coverage_inputs_are_separate(monkeypatch):
    calls = []

    def fake_request(client, name, schema, instructions, content):
        calls.append((name, content))
        return ({"claims": []} if name == "grounding_judgment" else {"key_points": []}), .01

    monkeypatch.setattr(ev, "_judge_request", fake_request)
    result = ev.judge_answer(object(), {
        "question": "What is FIrpic?", "expected_points": ["GOLD_ONLY_MARKER"],
    }, {"answer": "FIrpic is blue.", "evidence": [{
        "title": "Paper", "page": 1, "text": "EVIDENCE_ONLY_MARKER",
    }]})
    assert "GOLD_ONLY_MARKER" not in calls[0][1]
    assert "GOLD_ONLY_MARKER" in calls[1][1]
    assert "EVIDENCE_ONLY_MARKER" in calls[0][1]
    assert "EVIDENCE_ONLY_MARKER" not in calls[1][1]
    assert result["judge_policy"] == ev.JUDGE_POLICY
    assert result["judge_cost_usd"] == .02


def test_conversation_v2_preserves_inputs_and_historical_rubric():
    folder = Path(ev.PROJECT_ROOT) / "eval"
    old = json.loads((folder / "conversations.json").read_text())
    new = json.loads((folder / "conversations_v2.json").read_text())
    for before, after in zip(old["conversations"], new["conversations"]):
        assert before["id"] == after["id"]
        for a, b in zip(before["turns"], after["turns"]):
            assert a["question"] == b["question"]
            assert a["expected_mode"] == b["expected_mode"]
    old_turn = next(c for c in old["conversations"] if c["id"] == "clarify")["turns"][2]
    new_turn = next(c for c in new["conversations"] if c["id"] == "clarify")["turns"][2]
    assert "harvest triplet" in old_turn["reference_question"]
    assert new_turn["reference_question"] == "Compare thermally activated delayed fluorescence (TADF) with phosphorescence."
    assert Path(multi.DATASET).name == "conversations_v2.json"
    saved = {"manifest": {"questions_sha256": hashlib.sha256((folder / "conversations.json").read_bytes()).hexdigest()}}
    assert Path(ev.resolve_conversation_dataset(saved)).name == "conversations.json"
    with pytest.raises(ValueError, match="--dataset"):
        ev.resolve_conversation_dataset({"manifest": {"questions_sha256": "unknown"}})


def saved_output(multiturn=False):
    record = {
        "id": "clarify/3" if multiturn else "e01", "category": "normal", "pair_id": None,
        "question": "Phosphorescence." if multiturn else "What is an exciton?",
        "standalone_question": "Compare TADF with phosphorescence.",
        "answer": "FIrpic is blue.", "mode": "RAG", "expected_mode": "RAG",
        "used_files": [], "latency_s": 7, "cost_usd": .12, "run": 0,
        "searches": 1, "llm_calls": 3, "prompt_tokens": 10, "completion_tokens": 5,
        "evidence": [{"chunk_id": "c12345678", "title": "Paper", "text": "FIrpic is blue."}],
        "retrieved_evidence": [{"chunk_id": "c12345678", "text": "FIrpic is blue."}],
    }
    if multiturn:
        record.update(conversation_id="clarify", turn=3,
                      interpretation={"correct": False, "reason": "old rubric", "judge_cost_usd": .9})
    return {"engine": "graph", "label": "fixture", "created_at": "2026-10-09T19:00:00",
            "config": {"judge_policy": "scoped_absence_v2", "agent": {"workers": 2}},
            "manifest": {"judge_policy": "scoped_absence_v2", "code_sha256": {"original": "hash"}},
            "records": [record]}


@pytest.mark.parametrize("multiturn", [False, True])
def test_rejudge_saved_results_preserves_agent_run_and_refreshes_judges(tmp_path, monkeypatch, multiturn):
    public_dir, raw_dir = tmp_path / "results", tmp_path / "results" / "raw"
    raw_dir.mkdir(parents=True)
    monkeypatch.setattr(ev, "RESULTS_DIR", str(public_dir))
    monkeypatch.setattr(ev, "RAW_RESULTS_DIR", str(raw_dir))
    original = saved_output(multiturn)
    public_path, raw_path = public_dir / "input.json", raw_dir / "input.json"
    public_path.write_text(json.dumps(public_copy(original)))
    raw_path.write_text(json.dumps(original))
    original_public, original_raw = public_path.read_bytes(), raw_path.read_bytes()
    seen = []

    def fake_judge(client, item, record):
        seen.append(item)
        return {"claims": [{"claim": "FIrpic is blue.", "answer_quote": "FIrpic is blue.",
                            "verdict": "supported", "reason": "fixture"}],
                "key_points": [{"index": 0, "covered": True}],
                "judge_cost_usd": .02, "judge_policy": ev.JUDGE_POLICY}

    monkeypatch.setattr(ev, "OpenAI", lambda: object())
    monkeypatch.setattr(ev, "judge_answer", fake_judge)
    monkeypatch.setattr(multi, "grade_interpretation", lambda client, turn, record: {
        "correct": True, "reason": "fixture", "judge_cost_usd": .03,
    })
    monkeypatch.setattr(ev, "build_agent", lambda *args, **kwargs: pytest.fail("Agent must not run during rejudge"))
    ev.rejudge([str(public_path)], dataset_path=multi.DATASET if multiturn else None)
    result = json.loads(next(raw_dir.glob("graph_fixture_rejudged_*.json")).read_text())
    assert result["engine_manifest"] == original["manifest"]
    assert "manifest" not in result
    assert result["judge_manifest"]["judge_policy"] == result["config"]["judge_policy"] == ev.JUDGE_POLICY
    assert result["engine_created_at"] == original["created_at"]
    assert result["records"][0]["answer"] == original["records"][0]["answer"]
    assert result["records"][0]["latency_s"] == 7
    assert result["records"][0]["cost_usd"] == .12
    assert result["summary"]["grounding_evaluation_complete"] is True
    assert public_path.read_bytes() == original_public and raw_path.read_bytes() == original_raw
    exported = json.loads(next(public_dir.glob("graph_fixture_rejudged_*.json")).read_text())
    assert all("text" not in e for field in ("evidence", "retrieved_evidence")
               for e in exported["records"][0][field])
    if multiturn:
        assert seen[0]["id"] == "clarify/3"
        assert seen[0]["question"] == "Compare thermally activated delayed fluorescence (TADF) with phosphorescence."
        assert result["conversation_metrics"]["interpretation_accuracy"] == 1
        assert result["grounding_judge_cost_usd"] == .02
        assert result["interpretation_judge_cost_usd"] == .03
        assert result["judge_manifest"]["dataset_version"] == 2
        assert Path(ev.resolve_conversation_dataset(result)).name == "conversations_v2.json"


def test_rejudge_rejects_missing_raw_evidence_before_api_calls(tmp_path, monkeypatch):
    output = saved_output()
    output["records"][0]["evidence"][0].pop("text")
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps(output))
    monkeypatch.setattr(ev, "OpenAI", lambda: pytest.fail("Must validate before creating a paid client"))
    with pytest.raises(ValueError, match="Raw cited evidence"):
        ev.rejudge([str(path)])
