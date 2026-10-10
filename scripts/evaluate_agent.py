"""Evaluate the OLED agent on a fixed question set.

Each result includes:
- Mode accuracy, false rejection, and false acceptance.
- Unsupported claims, graded against cited chunks by an independent judge.
- Expected key-point coverage.
- Expected source-paper usage.
- Searches, LLM calls, latency, tokens, and estimated cost.

The independent judge is used only by this evaluation script, not by the application.

Usage (run from the project root, e.g. inside the Docker image)
    python scripts/evaluate_agent.py --run --label final
    python scripts/evaluate_agent.py --run --ids t01,mh1 --repeat 3
    AGENT_REVIEWER=false python scripts/evaluate_agent.py --run --label no_reviewer
    python scripts/evaluate_agent.py --compare eval/results/A.json eval/results/B.json [C.json ...] --out name.md
    python scripts/evaluate_agent.py --make-public eval/results/*.json
    python scripts/evaluate_agent.py --rejudge eval/results/raw/A.json [B.json ...]

Each run is saved twice.
`eval/results/raw/` keeps the full result, including evidence text, and is git-ignored because the papers are not ours to publish.
`eval/results/` keeps the same result without chunk text.

The active models and feature flags come from the same environment settings as the application and are recorded in each result file.
"""

import argparse
import json
import os
import statistics
import sys
import time
import hashlib
import platform
from importlib.metadata import version, PackageNotFoundError
from uuid import uuid4
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from openai import OpenAI  # noqa: E402

import config  # noqa: E402
from utils import estimate_cost_usd  # noqa: E402

QUESTIONS_FILE = os.path.join(PROJECT_ROOT, "eval", "questions.json")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "eval", "results")
# Raw results contain source text and stay git-ignored; public results omit it.
RAW_RESULTS_DIR = os.path.join(RESULTS_DIR, "raw")

JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gpt-5")
JUDGE_REASONING_EFFORT = os.getenv("EVAL_JUDGE_REASONING_EFFORT", "low")
JUDGE_THREADS = 4
REJECTION_MODES = {"NO_ANSWER_IN_DOCS", "OFF_TOPIC", "CLARIFICATION"}


# ================================
# Agent setup
# ================================
def build_agent(engine="graph", session_db=None):
    """Create the agent the same way the app does."""
    # Lazy imports keep reporting commands from loading retrieval models.
    from agent_runtime import AgentAssistant
    from retrieval import build_retriever

    if engine == "legacy":
        if session_db:
            raise ValueError("The legacy baseline has no session memory")
        return AgentAssistant(build_retriever())
    from agent_graph import GraphAgentAssistant
    retriever = build_retriever()
    return (GraphAgentAssistant.with_sqlite(retriever, session_db)
            if session_db else GraphAgentAssistant(retriever))


# ================================
# Running one question
# ================================
def run_agent(agent, question, thread_id=None):
    """Run one question and collect the agent's usage, evidence and trace.

    One request may use several models for routing, workers and review.
    Its cost therefore comes from the per-model usage reported by the agent.
    """
    result = agent.query(question, thread_id=thread_id) if thread_id else agent.query(question)
    report = result.get("agent", {})
    usage = report.get("usage", {})
    plan = report.get("plan") or {}
    return result, {
        "evidence": report.get("evidence", []),
        "retrieved_evidence": report.get("retrieved_evidence", []),
        "searches": report.get("searches", 0),
        "llm_calls": usage.get("llm_calls", 0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "cached_tokens": usage.get("cached_tokens", 0),
        "reasoning_tokens": usage.get("reasoning_tokens", 0),
        "usage_by_model": usage.get("by_model", {}),
        "calls_by_role": usage.get("calls_by_role", {}),
        "cost_usd": usage.get("cost_usd"),
        "complexity": plan.get("complexity"),
        "route": report.get("route"),
        "escalated": report.get("escalated", False),
        "review_rounds": report.get("review_rounds", 0),
        "parallel_turns": report.get("parallel_turns", 0),
        "stop_reason": report.get("stop_reason", ""),
        "trace": report.get("trace", []),
        "original_question": result.get("original_question", question),
        "standalone_question": result.get("standalone_question", question),
    }


def run_question(agent, item, thread_id=None):
    """Run one question and return a record with every raw measurement."""
    start = time.time()
    result, extra = run_agent(agent, item["question"], thread_id=thread_id)
    cost = extra.pop("cost_usd")
    latency = time.time() - start

    used_files = sorted({entry["file_name"] for entry in extra["evidence"]})
    return {
        "id": item["id"],
        "category": item["category"],
        "pair_id": item.get("pair_id"),
        "question": item["question"],
        "expected_mode": item["expected_mode"],
        "mode": result["mode"],
        "answer": result.get("answer") or "",
        "latency_s": round(latency, 2),
        "cost_usd": cost,
        "used_files": used_files,
        **extra,
    }


# ================================
# Independent judge
# ================================
GROUNDING_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "answer_quote": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["supported", "partially_supported", "unsupported"],
                    },
                    "reason": {"type": "string"},
                },
                "required": ["claim", "answer_quote", "verdict", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["claims"],
    "additionalProperties": False,
}

COVERAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "key_points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "covered": {"type": "boolean"},
                },
                "required": ["index", "covered"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["key_points"],
    "additionalProperties": False,
}

GROUNDING_INSTRUCTIONS = """You are a strict grounding grader for a document-grounded OLED assistant.

Split only the ANSWER into its atomic factual claims (at most 12; skip greetings,
hedges, and restatements of the question). For every claim, include an exact
contiguous quote from the ANSWER that expresses it. Never create a claim from
the question, expected answer, or evidence. Judge each claim ONLY against the
EVIDENCE passages, not against your own knowledge:
- supported: the evidence states it or directly implies it.
- partially_supported: the gist is in the evidence but a specific detail
  (number, material, condition, causal link) is not.
- unsupported: the evidence does not contain it, or contradicts it.
- A statement scoped to the supplied evidence (e.g. "the retrieved evidence
  does not state X") is supported when none of the EVIDENCE passages state X.
  A broader statement about the whole document collection remains unsupported,
  because only the supplied passages can be checked.

Return JSON only."""

COVERAGE_INSTRUCTIONS = """You are a strict answer-coverage grader.

For each EXPECTED POINT, mark covered=true only when the ANSWER conveys that
point. This task measures coverage only. Do not extract or judge factual claims.

Return JSON only."""

# Save the grading-policy version because rates from different policies are not directly comparable.
# `"strict_absence_v1"` allowed every absence statement to be marked unsupported.
# `"scoped_absence_v2"` supports an absence statement scoped to supplied evidence when that evidence lacks the stated information.
# `"separate_grounding_coverage_v3"` hides expected points from grounding and requires an answer quote for every extracted claim.
JUDGE_POLICY = "separate_grounding_coverage_v3"


def _judge_request(client, schema_name, schema, instructions, content):
    """Run one independent judge task and return its JSON plus call cost."""
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        reasoning_effort=JUDGE_REASONING_EFFORT,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        },
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": content},
        ],
    )
    result = json.loads(response.choices[0].message.content)
    usage = response.usage
    cost = estimate_cost_usd(JUDGE_MODEL, usage.prompt_tokens, usage.completion_tokens)
    return result, cost or 0.0


def judge_answer(client, item, record):
    """Grade grounding and expected-point coverage in separate model calls."""
    evidence_text = "\n\n".join(
        f"[E{index}] {entry['title']} (p.{entry.get('page')})\n{entry['text']}"
        for index, entry in enumerate(record["evidence"], 1)
    ) or "(no evidence)"
    points_text = "\n".join(
        f"{index}. {point}" for index, point in enumerate(item["expected_points"])
    ) or "(none)"

    grounding, grounding_cost = _judge_request(
        client,
        "grounding_judgment",
        GROUNDING_SCHEMA,
        GROUNDING_INSTRUCTIONS,
        f"QUESTION:\n{item['question']}\n\nANSWER:\n{record['answer']}\n\nEVIDENCE:\n{evidence_text}",
    )
    coverage, coverage_cost = _judge_request(
        client,
        "coverage_judgment",
        COVERAGE_SCHEMA,
        COVERAGE_INSTRUCTIONS,
        f"QUESTION:\n{item['question']}\n\nANSWER:\n{record['answer']}\n\nEXPECTED POINTS:\n{points_text}",
    )
    return {
        "claims": grounding["claims"],
        "key_points": coverage["key_points"],
        "judge_cost_usd": grounding_cost + coverage_cost,
        "judge_policy": JUDGE_POLICY,
    }


def add_scores(item, record, judgment):
    """Add mode, grounding, coverage and source metrics to one result record.

    `judgment` is `None` when the judge is disabled or the agent does not return a RAG answer.
    In that case, claim metrics remain empty.
    """
    expected, mode = record["expected_mode"], record["mode"]
    record["mode_correct"] = mode == expected
    record["false_rejection"] = expected == "RAG" and mode in REJECTION_MODES
    record["false_acceptance"] = expected != "RAG" and mode == "RAG"
    record["error"] = mode == "ERROR"

    expected_sources = [s.lower() for s in item.get("expected_sources", [])]
    record["source_hit"] = (
        any(any(s in f.lower() for s in expected_sources) for f in record["used_files"])
        if expected_sources and mode == "RAG"
        else None
    )

    if judgment is None:
        record.update(claims=[], claim_count=0, unsupported_claims=0,
                      partial_claims=0, claim_extraction_errors=[],
                      grounding_valid=False, key_points_covered=None, judge_cost_usd=0.0)
        return record

    answer_text = " ".join(record["answer"].split())
    valid_claims = []
    extraction_errors = []
    for claim in judgment["claims"]:
        quote = claim.get("answer_quote")
        normalized_quote = " ".join(quote.split()) if isinstance(quote, str) else ""
        legacy_claim = quote is None and judgment.get("judge_policy") != JUDGE_POLICY
        if legacy_claim or (normalized_quote and normalized_quote in answer_text):
            valid_claims.append(claim)
        else:
            extraction_errors.append(claim)

    verdicts = [claim["verdict"] for claim in valid_claims]
    covered = [point["covered"] for point in judgment["key_points"]]
    record.update(
        claims=valid_claims,
        claim_extraction_errors=extraction_errors,
        grounding_valid=bool(valid_claims) and not extraction_errors,
        claim_count=len(verdicts),
        unsupported_claims=verdicts.count("unsupported"),
        partial_claims=verdicts.count("partially_supported"),
        key_points_covered=(sum(covered) / len(covered)) if covered else None,
        judge_cost_usd=judgment.get("judge_cost_usd") or 0.0,
    )
    return record


# ================================
# Summaries
# ================================
def mean(values):
    """Average of the non-None values, or None if there are none."""
    values = [value for value in values if value is not None]
    return round(statistics.mean(values), 4) if values else None


def percentile(values, share):
    """Simple nearest-rank percentile, with share between 0 and 1 (e.g. 0.9 for p90)."""
    values = sorted(values)
    if not values:
        return None
    index = min(len(values) - 1, max(0, round(share * (len(values) - 1))))
    return values[index]


def summarize(records):
    """Aggregate metrics over a list of records."""
    answerable = [r for r in records if r["expected_mode"] == "RAG"]
    unanswerable = [r for r in records if r["expected_mode"] != "RAG"]
    answered = [r for r in records if r["mode"] == "RAG"]
    claims = sum(r["claim_count"] for r in answered)
    invalid_grounding = sum(
        not r.get("grounding_valid", bool(r["claim_count"]) and not r.get("claim_extraction_errors"))
        for r in answered
    )
    latencies = [r["latency_s"] for r in records]

    return {
        "runs": len(records),
        "mode_accuracy": mean([1.0 if r["mode_correct"] else 0.0 for r in records]),
        "false_rejection_rate": mean([1.0 if r["false_rejection"] else 0.0 for r in answerable]),
        "false_acceptance_rate": mean([1.0 if r["false_acceptance"] else 0.0 for r in unanswerable]),
        "error_count": sum(1 for r in records if r["error"]),
        "unsupported_claim_rate": round(sum(r["unsupported_claims"] for r in answered) / claims, 4) if claims else None,
        "partial_claim_rate": round(sum(r["partial_claims"] for r in answered) / claims, 4) if claims else None,
        "answers_with_unsupported": sum(1 for r in answered if r["unsupported_claims"] > 0),
        "judge_extraction_errors": sum(len(r.get("claim_extraction_errors", [])) for r in answered),
        "grounding_invalid_answers": invalid_grounding,
        "grounding_evaluation_complete": invalid_grounding == 0 if answered else None,
        "answered_count": len(answered),
        "key_point_coverage": mean([r["key_points_covered"] for r in answered]),
        "source_hit_rate": mean([1.0 if r["source_hit"] else 0.0 for r in answered if r["source_hit"] is not None]),
        "avg_searches": mean([r["searches"] for r in records]),
        "avg_llm_calls": mean([r["llm_calls"] for r in records]),
        "latency_mean_s": mean(latencies),
        "latency_p50_s": percentile(latencies, 0.5),
        "latency_p90_s": percentile(latencies, 0.9),
        "avg_prompt_tokens": mean([r["prompt_tokens"] for r in records]),
        "avg_completion_tokens": mean([r["completion_tokens"] for r in records]),
        "total_cost_usd": round(sum(r["cost_usd"] or 0.0 for r in records), 4),
        "avg_cost_usd": mean([r["cost_usd"] for r in records]),
        # Runs that used a model with no price in MODEL_PRICING_PER_1M.
        "unpriced_runs": sum(1 for r in records if r["cost_usd"] is None),
        "tokens_by_model": tokens_by_model(records),
        "variant_only_rejections": sum(
            pair["variant_only_rejections"] for pair in pair_report(records).values()
        ),
        # Result files from the first agent version lack some fields below, so read them with `.get()`.
        "stop_reasons": dict(Counter(r.get("stop_reason") or "n/a" for r in records)),
        "routes": dict(Counter(r.get("route") or "n/a" for r in records)),
        "escalations": sum(1 for r in records if r.get("escalated")),
        "runs_with_parallel_calls": sum(1 for r in records if r.get("parallel_turns")),
        "avg_review_rounds": mean([r.get("review_rounds") for r in records if "review_rounds" in r]),
    }


def cost_from_usage(usage_by_model):
    """Calculate one run's USD cost from per-model token counts.

    Return `None` when any model used by the run has no configured price.
    """
    total = 0.0
    for model, usage in usage_by_model.items():
        cost = estimate_cost_usd(
            model, usage["input_tokens"], usage["output_tokens"], usage.get("cached_tokens", 0)
        )
        if cost is None:
            return None
        total += cost
    return round(total, 6)


def tokens_by_model(records):
    """Add up the input/output tokens per model, so we can still price unpriced models later."""
    totals = {}
    for record in records:
        for model, usage in (record.get("usage_by_model") or {}).items():
            entry = totals.setdefault(model, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0})
            for key in entry:
                entry[key] += usage.get(key, 0)
    return totals


def summarize_by_category(records):
    """Summary per question category (normal, typo, multi_hop, ...)."""
    categories = sorted({r["category"] for r in records})
    return {c: summarize([r for r in records if r["category"] == c]) for c in categories}


def pair_report(records):
    """Compare normally worded questions with their variants.

    A variant-only rejection occurs when the normal question is answered but its typo, paraphrase, or abbreviation is rejected.
    """
    report = {}
    pairs = sorted({r["pair_id"] for r in records if r["pair_id"]})
    for pair_id in pairs:
        members = [r for r in records if r["pair_id"] == pair_id]
        normal_answered = any(r["mode"] == "RAG" for r in members if r["category"] == "normal")
        variants = [r for r in members if r["category"] != "normal"]
        report[pair_id] = {
            "normal_answered": normal_answered,
            "variant_modes": {r["id"]: r["mode"] for r in variants},
            "variant_only_rejections": sum(
                1 for r in variants if normal_answered and r["mode"] in REJECTION_MODES
            ),
        }
    return report


# ================================
# Printing
# ================================
SUMMARY_ROWS = [
    ("mode_accuracy", "Mode accuracy"),
    ("false_rejection_rate", "False rejection rate (expected RAG)"),
    ("false_acceptance_rate", "False acceptance rate (expected rejection)"),
    ("error_count", "Errors"),
    ("unsupported_claim_rate", "Unsupported claim rate"),
    ("partial_claim_rate", "Partially supported claim rate"),
    ("answers_with_unsupported", "Answers with >=1 unsupported claim"),
    ("judge_extraction_errors", "Judge claim-extraction errors"),
    ("grounding_invalid_answers", "Answers without valid grounding evaluation"),
    ("grounding_evaluation_complete", "Grounding evaluation complete"),
    ("answered_count", "Answered (RAG)"),
    ("key_point_coverage", "Key-point coverage"),
    ("source_hit_rate", "Expected-source hit rate"),
    ("avg_searches", "Avg searches"),
    ("avg_llm_calls", "Avg LLM calls"),
    ("latency_mean_s", "Latency mean (s)"),
    ("latency_p50_s", "Latency p50 (s)"),
    ("latency_p90_s", "Latency p90 (s)"),
    ("avg_prompt_tokens", "Avg prompt tokens"),
    ("avg_completion_tokens", "Avg completion tokens"),
    ("avg_cost_usd", "Avg cost per question (USD)"),
    ("total_cost_usd", "Total cost (USD)"),
    ("unpriced_runs", "Runs with an unpriced model"),
    ("variant_only_rejections", "Variant-only rejections (pairs)"),
    ("stop_reasons", "Stop reasons"),
    ("routes", "Routes"),
    ("escalations", "Escalations (light -> heavy)"),
    ("runs_with_parallel_calls", "Runs with parallel tool calls"),
    ("avg_review_rounds", "Avg review rounds"),
]


def print_summary(title, summary):
    """Print one summary as an aligned two-column table."""
    print(f"\n=== {title} ===")
    for key, label in SUMMARY_ROWS:
        print(f"  {label:45s} {summary.get(key)}")


def print_records(records):
    """Print one line per run with the id, the expected and actual mode, and the key numbers."""
    print("\nid     category      expected           actual             route  srch llm  lat(s)  unsup/claims")
    for r in records:
        marker = "" if r["mode_correct"] else "  <-- mismatch"
        route = (r.get("route") or "-") + ("^" if r.get("escalated") else "")
        print(
            f"{r['id']:6s} {r['category']:13s} {r['expected_mode']:18s} {r['mode']:18s} "
            f"{route:6s} {r['searches']:4} {r['llm_calls']:4} {r['latency_s']:7.1f}  "
            f"{r['unsupported_claims']}/{r['claim_count']}{marker}"
        )


# ================================
# Commands
# ================================
def evaluate(args):
    """Run the selected questions and save raw and public result files.

    `args` contains the selected engine, question IDs, repeat count, label, session database and judge setting.
    """
    questions_path = args.questions or QUESTIONS_FILE
    with open(questions_path, encoding="utf-8") as handle:
        questions = json.load(handle)["questions"]
    if args.ids:
        wanted = set(args.ids.split(","))
        questions = [q for q in questions if q["id"] in wanted]

    agent = build_agent(args.engine, args.session_db)
    client = None if args.no_judge else OpenAI()

    # Judge calls wait mostly on the API, so they run in background threads while agent questions remain sequential.
    # Sequential agent execution keeps one question from affecting another question's latency.
    pending = []  # (item, record, future or None), in question order
    with ThreadPoolExecutor(max_workers=JUDGE_THREADS) as judge_pool:
        for item in questions:
            for run_index in range(args.repeat):
                # A fresh thread prevents evaluation questions from sharing history.
                thread_id = str(uuid4()) if args.session_db else None
                record = run_question(agent, item, thread_id=thread_id)
                record["run"] = run_index
                future = None
                if client and record["mode"] == "RAG" and record["answer"]:
                    future = judge_pool.submit(judge_answer, client, item, record)
                pending.append((item, record, future))
                print(f"[{item['id']} run {run_index}] {record['mode']:18s} {record['latency_s']:.1f}s", flush=True)

        records = []
        for item, record, future in pending:
            judgment = future.result() if future else None
            records.append(add_scores(item, record, judgment))

    output = {
        # Kept as a field so --compare can still read older result files.
        "engine": args.engine,
        "label": args.label,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": evaluation_manifest(questions_path),
        "repeat": args.repeat,
        "history_policy": "fresh_thread_per_question" if args.session_db else "stateless",
        "config": {
            "judge_model": None if args.no_judge else JUDGE_MODEL,
            "judge_policy": None if args.no_judge else JUDGE_POLICY,
            "candidate_top_k": config.CANDIDATE_TOP_K,
            "final_top_n": config.FINAL_TOP_N,
            "min_doc_relevance": config.MIN_DOC_RELEVANCE,
            # Models, feature flags, and budgets the agent actually ran with.
            "agent": agent.features(),
        },
        "summary": summarize(records),
        "by_category": summarize_by_category(records),
        "pairs": pair_report(records),
        "judge_cost_usd": round(sum(r["judge_cost_usd"] for r in records), 4),
        "records": records,
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"agent_{args.label + '_' if args.label else ''}{stamp}.json"
    raw_path = os.path.join(RAW_RESULTS_DIR, name)
    write_json(raw_path, output)
    path = os.path.join(RESULTS_DIR, name)
    write_json(path, public_copy(output))

    print_records(records)
    print_summary("agent overall", output["summary"])
    for category, summary in output["by_category"].items():
        print(f"  [{category}] acc={summary['mode_accuracy']} false_rej={summary['false_rejection_rate']} "
              f"unsup={summary['unsupported_claim_rate']} lat={summary['latency_mean_s']}")
    print("\nPairs:", json.dumps(output["pairs"], indent=1))
    print(f"\nJudge cost (not part of agent cost): ${output['judge_cost_usd']}")
    print(f"Saved {path} (public, no chunk text) and {raw_path} (local only)")
    if hasattr(agent, "close"):
        agent.close()


def evaluation_manifest(questions_path):
    """Record actual code/data/settings; a git revision alone misses local edits."""
    def digest(path):
        if not os.path.isfile(path):
            return None
        h = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
    src = os.path.join(PROJECT_ROOT, "src")
    packages = {}
    for name in ("langgraph", "langgraph-checkpoint-sqlite", "langchain-core", "langchain-community",
                 "langchain-text-splitters", "chromadb", "sentence-transformers", "openai"):
        try: packages[name] = version(name)
        except PackageNotFoundError: packages[name] = None
    return {"python": platform.python_version(), "packages": packages,
            "code_sha256": {name: digest(os.path.join(src, name)) for name in sorted(os.listdir(src)) if name.endswith(".py")},
            "questions_sha256": digest(questions_path),
            "corpus_sqlite_sha256": digest(os.path.join(config.DB_PATH, "chroma.sqlite3")),
            "judge_policy": JUDGE_POLICY, "judge_model": JUDGE_MODEL,
            "judge_reasoning_effort": JUDGE_REASONING_EFFORT,
            "reasoning_effort": config.AGENT_REASONING_EFFORT or config.MODEL_REASONING_EFFORT,
            "embedding": config.EMBEDDING_MODEL, "reranker": config.RERANKER_MODEL,
            "reranker_enabled": config.RERANKER_ENABLED,
            "sigmoid": [config.SIGMOID_MIDPOINT, config.SIGMOID_STEEPNESS]}


# ================================
# Public export
# ================================
def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, default=str)


def public_copy(output):
    """Return a publishable copy without source-paper text.

    Chunk IDs, titles, URLs, file names and pages remain so citations can still be traced to their papers.
    """
    public = json.loads(json.dumps(output, default=str))
    for record in public.get("records", []):
        for field in ("evidence", "retrieved_evidence"):
            record[field] = [
                {key: value for key, value in entry.items() if key != "text"}
                for entry in record.get(field, [])
            ]
    public["chunk_text_removed"] = True
    return public


def make_public(paths):
    """Back up result files and write public copies.

    The operation is safe to repeat.
    Files already marked public are skipped, and existing raw backups are never overwritten.
    """
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            output = json.load(handle)
        name = os.path.basename(path)
        raw_path = os.path.join(RAW_RESULTS_DIR, name)
        if output.get("chunk_text_removed"):
            print(f"{name}: already public, skipped (raw backup left untouched)")
            continue
        if os.path.exists(raw_path):
            print(f"{name}: raw backup exists, kept as is")
        else:
            write_json(raw_path, output)
        write_json(os.path.join(RESULTS_DIR, name), public_copy(output))
        print(f"{name}: public copy written, original in {raw_path}")


def resolve_conversation_dataset(output, dataset_path=None):
    """Reuse the recorded rubric by hash; a changed rubric needs an explicit path."""
    if dataset_path:
        return str(Path(dataset_path))
    judge_manifest = output.get("judge_manifest", {})
    engine_manifest = output.get("engine_manifest", output.get("manifest", {}))
    expected = judge_manifest.get("dataset_sha256") or engine_manifest.get("questions_sha256")
    for path in sorted(Path(PROJECT_ROOT, "eval").glob("conversations*.json")):
        if hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            return str(path)
    raise ValueError("Conversation dataset does not match a saved hash; supply --dataset explicitly")


def judge_manifest(questions_path, dataset_path=None):
    """Describe only the current grading, never relabel the original agent run."""
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "judge_policy": JUDGE_POLICY, "judge_model": JUDGE_MODEL,
        "judge_reasoning_effort": JUDGE_REASONING_EFFORT,
        "questions_file": Path(questions_path).name,
        "questions_sha256": digest(questions_path),
        "code_sha256": {name: digest(Path(__file__).with_name(name))
                        for name in ("evaluate_agent.py", "evaluate_multiturn.py")},
    }
    if dataset_path:
        dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
        manifest.update(dataset_file=Path(dataset_path).name,
                        dataset_sha256=digest(dataset_path),
                        dataset_version=dataset.get("version", 1))
    return manifest


def rejudge(paths, questions_path=None, dataset_path=None):
    """Rejudge saved answers without rerunning the agent.

    Raw results already contain the original answers and cited evidence, so a new judge policy can be applied without repeating agent calls.
    Public inputs use the corresponding raw file.
    Multi-turn results use the recorded conversation-rubric hash unless `dataset_path` explicitly overrides it.
    """
    questions_path = questions_path or QUESTIONS_FILE
    with open(questions_path, encoding="utf-8") as handle:
        questions = {item["id"]: item for item in json.load(handle)["questions"]}

    for path in paths:
        with open(path, encoding="utf-8") as handle:
            output = json.load(handle)

        # Public files have no paper text, so load their local raw copy.
        if output.get("chunk_text_removed"):
            raw_path = os.path.join(RAW_RESULTS_DIR, os.path.basename(path))
            if not os.path.exists(raw_path):
                raise FileNotFoundError(f"Raw result required for rejudging: {raw_path}")
            with open(raw_path, encoding="utf-8") as handle:
                output = json.load(handle)

        is_conversation = "conversation_metrics" in output or any(
            "conversation_id" in record for record in output["records"]
        )
        rubric_path = None
        items = questions
        if is_conversation:
            from evaluate_multiturn import build_turn_item, conversation_metrics, grade_interpretation

            rubric_path = resolve_conversation_dataset(output, dataset_path)
            with open(rubric_path, encoding="utf-8") as handle:
                conversations = json.load(handle)["conversations"]
            items = {}
            for conversation in conversations:
                for index in range(len(conversation["turns"])):
                    item = build_turn_item(conversation, index, questions)
                    items[item["id"]] = item
        elif dataset_path:
            raise ValueError("--dataset is only valid for conversation results")

        # Check all inputs before paying for a judge call.
        for record in output["records"]:
            if record["id"] not in items:
                raise ValueError(f"No grading item for {record['id']}; check --questions/--dataset")
            if record["mode"] == "RAG" and (
                not record.get("evidence") or
                any(not isinstance(e.get("text"), str) or not e["text"].strip()
                    for e in record["evidence"])
            ):
                raise ValueError(f"Raw cited evidence required for {record['id']}")

        current_judge_manifest = judge_manifest(questions_path, rubric_path)
        client = OpenAI()
        pending = []
        with ThreadPoolExecutor(max_workers=JUDGE_THREADS) as judge_pool:
            for record in output["records"]:
                item = items[record["id"]]
                record["expected_mode"] = item["expected_mode"]
                future = None
                if record["mode"] == "RAG" and record.get("answer"):
                    grounding_item = {**item, "question": item.get("reference_question", item["question"])}
                    future = judge_pool.submit(judge_answer, client, grounding_item, record)
                interpretation = (judge_pool.submit(grade_interpretation, client, item, record)
                                  if is_conversation else None)
                pending.append((item, record, future, interpretation))

            records = []
            for item, record, future, interpretation in pending:
                add_scores(item, record, future.result() if future else None)
                if interpretation is not None:
                    record["interpretation"] = interpretation.result()
                records.append(record)

        original_label = output.get("label") or "run"
        output["engine_created_at"] = output.get("engine_created_at", output.get("created_at"))
        output["created_at"] = datetime.now().isoformat(timespec="seconds")
        output["label"] = f"{original_label}_rejudged"
        output.setdefault("config", {})["judge_model"] = JUDGE_MODEL
        output["config"]["judge_policy"] = JUDGE_POLICY
        output["config"]["judge_reasoning_effort"] = JUDGE_REASONING_EFFORT
        output.setdefault("engine_manifest", output.get("manifest", {}))
        output.pop("manifest", None)
        output["judge_manifest"] = current_judge_manifest
        output["summary"] = summarize(records)
        output["by_category"] = summarize_by_category(records)
        output["pairs"] = pair_report(records)
        output["judge_cost_usd"] = round(sum(r["judge_cost_usd"] for r in records), 4)
        output["records"] = records
        if is_conversation:
            output["conversation_metrics"] = conversation_metrics(records)
            output["grounding_judge_cost_usd"] = output["judge_cost_usd"]
            output["interpretation_judge_cost_usd"] = sum(
                r["interpretation"].get("judge_cost_usd") or 0 for r in records
            )
        output["evaluation_note"] = (
            "Rejudged saved answers; the agent was not rerun. engine_manifest describes "
            "the original execution; judge_manifest describes this grading and rubric."
        )

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{output['engine']}_{output['label']}_{stamp}.json"
        raw_path = os.path.join(RAW_RESULTS_DIR, name)
        public_path = os.path.join(RESULTS_DIR, name)
        write_json(raw_path, output)
        write_json(public_path, public_copy(output))
        print_summary(output["label"], output["summary"])
        print(f"Saved {public_path} (public, no chunk text) and {raw_path} (local only)")


def compare(paths, output_name="comparison.md"):
    """Print and save a side-by-side comparison of two or more result files.

    The Markdown report is written to `eval/results/<output_name>`.
    """
    runs = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            runs.append(json.load(handle))
    names = [run["engine"] + (f" ({run['label']})" if run.get("label") else "") for run in runs]

    # Recompute summaries so older files use the current metrics and prices.
    for run in runs:
        for record in run["records"]:
            if record.get("usage_by_model"):
                record["cost_usd"] = cost_from_usage(record["usage_by_model"])
        run["summary"] = summarize(run["records"])
        run["by_category"] = summarize_by_category(run["records"])

    def table_header(first_column):
        return [
            f"| {first_column} | " + " | ".join(names) + " |",
            "|---|" + "---|" * len(names),
        ]

    policies = {run.get("config", {}).get("judge_policy") for run in runs}
    lines = []
    if len(policies) > 1:
        lines += ["> Judge policies differ. Claim-grounding rates are not directly comparable; rejudge under one policy.", ""]
    lines += table_header("Metric")
    for key, label in SUMMARY_ROWS:
        lines.append(f"| {label} | " + " | ".join(str(run["summary"].get(key)) for run in runs) + " |")

    lines += ["", "Per category: mode accuracy / false rejection / unsupported claim rate / mean latency", ""]
    lines += table_header("Category")
    categories = sorted(set().union(*(run["by_category"] for run in runs)))
    for category in categories:
        cells = []
        for run in runs:
            s = run["by_category"].get(category, {})
            cells.append(
                f"{s.get('mode_accuracy')} / {s.get('false_rejection_rate')} / "
                f"{s.get('unsupported_claim_rate')} / {s.get('latency_mean_s')}"
            )
        lines.append(f"| {category} | " + " | ".join(cells) + " |")

    # Compare the first run of each question across configurations.
    firsts = [{r["id"]: r for r in run["records"] if r.get("run", 0) == 0} for run in runs]
    common_ids = [qid for qid in firsts[0] if all(qid in f for f in firsts)]
    lines += ["", "Questions whose mode differs:", ""]
    lines += [
        "| id | expected | " + " | ".join(names) + " |",
        "|---|---|" + "---|" * len(names),
    ]
    for qid in common_ids:
        modes = [f[qid]["mode"] for f in firsts]
        if len(set(modes)) > 1:
            lines.append(f"| {qid} | {firsts[0][qid]['expected_mode']} | " + " | ".join(modes) + " |")

    report = "\n".join(lines)
    print(report)
    path = os.path.join(RESULTS_DIR, output_name)
    with open(path, "w", encoding="utf-8") as handle:
        files = "\n".join(f"- {name}: `{os.path.basename(p)}`" for name, p in zip(names, paths))
        handle.write(f"# Agent configuration comparison\n\n{files}\n\n{report}\n")
    print(f"\nSaved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true", help="Run the agent on the question set (paid API calls)")
    parser.add_argument("--engine", choices=("legacy", "graph"), default="graph")
    parser.add_argument("--questions", help="Question-set JSON; default: eval/questions.json")
    parser.add_argument("--dataset", help="Conversation rubric override for --rejudge; default: match the saved dataset hash")
    parser.add_argument("--session-db", help="SQLite checkpointer path; use a fresh thread per question")
    parser.add_argument("--label", default="", help="Tag added to the results file name")
    parser.add_argument("--ids", default="", help="Comma-separated question ids to run")
    parser.add_argument("--repeat", type=int, default=1, help="Runs per question (stochastic checks)")
    parser.add_argument("--no-judge", action="store_true", help="Skip the claim-grounding judge")
    parser.add_argument("--compare", nargs="+", metavar="RESULT_JSON", help="Two or more result files")
    parser.add_argument("--out", default="comparison.md", help="File name for --compare output (in eval/results)")
    parser.add_argument("--make-public", nargs="+", metavar="RESULT_JSON",
                        help="Strip chunk text from existing result files (originals go to eval/results/raw/)")
    parser.add_argument("--rejudge", nargs="+", metavar="RESULT_JSON",
                        help="Re-run the judge on saved results without re-running the agent")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")
    if args.dataset and not args.rejudge:
        parser.error("--dataset requires --rejudge")

    if args.make_public:
        make_public(args.make_public)
    elif args.rejudge:
        rejudge(args.rejudge, args.questions, args.dataset)
    elif args.compare:
        if len(args.compare) < 2:
            parser.error("--compare needs at least two result files.")
        compare(args.compare, args.out)
    elif args.run:
        evaluate(args)
    else:
        parser.error("Use --run, --compare, --make-public, or --rejudge.")


if __name__ == "__main__":
    main()
