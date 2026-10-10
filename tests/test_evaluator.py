"""Checks for evaluation separation and public-result redaction."""

from evaluate_agent import add_scores, public_copy


def test_claim_without_answer_quote_is_excluded_from_metrics():
    """Expected-point leakage must be visible without lowering grounding scores."""
    item = {"expected_sources": []}
    record = {
        "answer": "FIrpic is a blue emitter.",
        "expected_mode": "RAG",
        "mode": "RAG",
        "used_files": [],
    }
    judgment = {
        "claims": [
            {
                "claim": "FIrpic is a blue emitter.",
                "answer_quote": "FIrpic is a blue emitter.",
                "verdict": "supported",
                "reason": "The evidence supports it.",
            },
            {
                "claim": "The lifetime improves tenfold.",
                "answer_quote": "The lifetime improves tenfold.",
                "verdict": "unsupported",
                "reason": "This came from the expected points.",
            },
        ],
        "key_points": [],
        "judge_cost_usd": 0.0,
    }

    scored = add_scores(item, record, judgment)

    assert scored["claim_count"] == 1
    assert scored["unsupported_claims"] == 0
    assert len(scored["claim_extraction_errors"]) == 1


def test_public_copy_removes_all_evidence_text():
    """Public exports keep identifiers but remove cited and retrieved text."""
    output = {
        "records": [
            {
                "evidence": [{"chunk_id": "c1", "text": "private cited text"}],
                "retrieved_evidence": [
                    {"chunk_id": "c2", "text": "private retrieved text"}
                ],
            }
        ]
    }

    exported = public_copy(output)

    assert exported["records"][0]["evidence"] == [{"chunk_id": "c1"}]
    assert exported["records"][0]["retrieved_evidence"] == [{"chunk_id": "c2"}]
