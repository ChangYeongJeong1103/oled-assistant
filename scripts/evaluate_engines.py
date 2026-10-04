"""
Evaluate the workflow and agent engines on the same question set.

What we measure for every question
----------------------------------
- Mode: we compare the engine's mode with the expected mode, which gives us
  the false rejections and false acceptances.
- Unsupported claims: an independent judge model splits the answer into
  claims and checks each one against the evidence the engine actually used
  (for the workflow its final documents, for the agent the chunks it cited).
- Key-point coverage: does the answer contain the expected facts?
- Source hit: did the engine use any of the expected source papers?
- Searches, LLM calls, latency, tokens, and estimated cost.

The judge is only part of this evaluation. It never runs inside the app.

Usage (run from the project root, e.g. inside the Docker image)
-----
    python scripts/evaluate_engines.py --engine workflow --label baseline
    python scripts/evaluate_engines.py --engine agent --ids t01,mh1 --repeat 3
    AGENT_ROUTING=true python scripts/evaluate_engines.py --engine agent --label routing
    python scripts/evaluate_engines.py --compare eval/results/A.json eval/results/B.json [C.json ...] --out name.md
    python scripts/evaluate_engines.py --make-public eval/results/*.json
    python scripts/evaluate_engines.py --rejudge eval/results/raw/A.json [B.json ...]

Each run is saved twice:
- eval/results/raw/ keeps the full result, including the text of the evidence
  chunks. This folder is git-ignored because the papers are not ours to
  publish.
- eval/results/ gets the same result without the chunk text.

The agent features (models, parallel search, routing, workers, reviewer) are
read from the environment exactly as in the app, and we save them with each
result file.
"""

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from openai import OpenAI  # noqa: E402

import config  # noqa: E402
from source_registry import describe_source  # noqa: E402
from utils import estimate_cost_usd  # noqa: E402

QUESTIONS_FILE = os.path.join(PROJECT_ROOT, "eval", "questions.json")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "eval", "results")
# Full results, including the text of every evidence chunk from the source
# papers. This folder is git-ignored. We only publish the copies in
# RESULTS_DIR, which have no chunk text.
RAW_RESULTS_DIR = os.path.join(RESULTS_DIR, "raw")

JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gpt-5")
JUDGE_REASONING_EFFORT = os.getenv("EVAL_JUDGE_REASONING_EFFORT", "low")
JUDGE_THREADS = 4
REJECTION_MODES = {"NO_ANSWER_IN_DOCS", "OFF_TOPIC"}


# ================================
# Engine setup
# ================================
def build_engine(engine_name):
    """Create the requested engine the same way the app does."""
    from rag_engine import StrictRAGAssistant, create_embeddings, get_vectorstore

    workflow = StrictRAGAssistant(
        vectorstore=get_vectorstore(create_embeddings()),
        llm_model=config.LLM_MODEL,
        off_topic_threshold=config.OFF_TOPIC_THRESHOLD,
        candidate_top_k=config.CANDIDATE_TOP_K,
        min_doc_relevance=config.MIN_DOC_RELEVANCE,
        final_top_n=config.FINAL_TOP_N,
        reranker_enabled=config.RERANKER_ENABLED,
        reranker_model=config.RERANKER_MODEL,
        temperature=config.LLM_TEMPERATURE,
        sigmoid_midpoint=config.SIGMOID_MIDPOINT,
        sigmoid_steepness=config.SIGMOID_STEEPNESS,
    )
    if engine_name == "workflow":
        return workflow

    # Import the agent lazily so we can run the workflow baseline even before
    # the agent exists.
    from agent_runtime import AgentAssistant

    return AgentAssistant(workflow)


# ================================
# Running one question
# ================================
def run_workflow(engine, question):
    """
    Run the fixed workflow pipeline and collect token usage with LangChain's
    OpenAI callback.

    The workflow sends its final documents to the LLM, so those are the
    documents we judge its answer against.
    """
    from langchain_community.callbacks.manager import get_openai_callback

    with get_openai_callback() as callback:
        result = engine.query(question)

    evidence = [
        {"text": doc.page_content, **describe_source(doc.metadata)}
        for doc in result.get("retrieved_docs", [])
    ]
    llm_called = callback.total_tokens > 0
    return result, {
        "evidence": evidence,
        "searches": 1,
        "llm_calls": 1 if llm_called else 0,
        "prompt_tokens": callback.prompt_tokens,
        "completion_tokens": callback.completion_tokens,
        "cached_tokens": 0,
        "stop_reason": result["mode"].lower(),
        "trace": [],
    }


def run_agent(engine, question):
    """
    Run the agent, which reports its own usage, evidence, and trace.

    The agent may use several models in one request (routing, workers,
    reviewer), so we take its cost from the per-model usage it reports.
    """
    result = engine.query(question)
    agent = result.get("agent", {})
    usage = agent.get("usage", {})
    plan = agent.get("plan") or {}
    return result, {
        "evidence": agent.get("evidence", []),
        "searches": agent.get("searches", 0),
        "llm_calls": usage.get("llm_calls", 0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "cached_tokens": usage.get("cached_tokens", 0),
        "reasoning_tokens": usage.get("reasoning_tokens", 0),
        "usage_by_model": usage.get("by_model", {}),
        "calls_by_role": usage.get("calls_by_role", {}),
        "cost_usd": usage.get("cost_usd"),
        "complexity": plan.get("complexity"),
        "route": agent.get("route"),
        "escalated": agent.get("escalated", False),
        "review_rounds": agent.get("review_rounds", 0),
        "parallel_turns": agent.get("parallel_turns", 0),
        "stop_reason": agent.get("stop_reason", ""),
        "trace": agent.get("trace", []),
    }


def run_question(engine_name, engine, item):
    """Run one question and return a record with every raw measurement."""
    start = time.time()
    if engine_name == "workflow":
        result, extra = run_workflow(engine, item["question"])
        # The workflow makes at most one LLM call, and always with LLM_MODEL,
        # so we can price it directly.
        cost = estimate_cost_usd(
            config.LLM_MODEL,
            extra["prompt_tokens"],
            extra["completion_tokens"],
            extra["cached_tokens"],
        )
    else:
        result, extra = run_agent(engine, item["question"])
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
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["supported", "partially_supported", "unsupported"],
                    },
                    "reason": {"type": "string"},
                },
                "required": ["claim", "verdict", "reason"],
                "additionalProperties": False,
            },
        },
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
    "required": ["claims", "key_points"],
    "additionalProperties": False,
}

JUDGE_INSTRUCTIONS = """You are a strict grader for a document-grounded OLED assistant.

Task 1 - Grounding. Split the ANSWER into its atomic factual claims (at most 12;
skip greetings, hedges, and restatements of the question). Judge each claim
ONLY against the EVIDENCE passages, not against your own knowledge:
- supported: the evidence states it or directly implies it.
- partially_supported: the gist is in the evidence but a specific detail
  (number, material, condition, causal link) is not.
- unsupported: the evidence does not contain it, or contradicts it.
- A statement scoped to the supplied evidence (e.g. "the retrieved evidence
  does not state X") is supported when none of the EVIDENCE passages state X.
  A broader statement about the whole document collection remains unsupported,
  because only the supplied passages can be checked.

Task 2 - Coverage. For each EXPECTED POINT (by index), mark covered=true when
the ANSWER conveys that point, regardless of the evidence.

Return JSON only."""

# Version of the grading rules above. We save it in every result file because
# rates graded under different policies can't be compared directly.
# "strict_absence_v1": earlier runs, where every absence statement could be
#                      marked unsupported.
# "scoped_absence_v2": a statement scoped to the supplied evidence is supported
#                      when the evidence indeed lacks it.
JUDGE_POLICY = "scoped_absence_v2"


def judge_answer(client, item, record):
    """
    Ask the judge model to grade the answer's claims and key-point coverage.

    The judge only sees the evidence the engine actually used, so each claim
    is graded against that evidence and nothing else.

    Args:
        client: OpenAI client used for the judge call
        item: Question entry from eval/questions.json (question, expected_points)
        record: Result of run_question() for this item (answer, evidence)

    Returns:
        dict: The judge's "claims" and "key_points", plus "judge_cost_usd"
    """
    evidence_text = "\n\n".join(
        f"[E{index}] {entry['title']} (p.{entry.get('page')})\n{entry['text']}"
        for index, entry in enumerate(record["evidence"], 1)
    ) or "(no evidence)"
    points_text = "\n".join(
        f"{index}. {point}" for index, point in enumerate(item["expected_points"])
    ) or "(none)"

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        reasoning_effort=JUDGE_REASONING_EFFORT,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "grounding_judgment", "strict": True, "schema": JUDGE_SCHEMA},
        },
        messages=[
            {"role": "system", "content": JUDGE_INSTRUCTIONS},
            {
                "role": "user",
                "content": (
                    f"QUESTION:\n{item['question']}\n\n"
                    f"ANSWER:\n{record['answer']}\n\n"
                    f"EXPECTED POINTS:\n{points_text}\n\n"
                    f"EVIDENCE:\n{evidence_text}"
                ),
            },
        ],
    )
    judgment = json.loads(response.choices[0].message.content)
    usage = response.usage
    judgment["judge_cost_usd"] = estimate_cost_usd(
        JUDGE_MODEL, usage.prompt_tokens, usage.completion_tokens
    )
    return judgment


def add_scores(item, record, judgment):
    """
    Turn the raw results and the judgment into per-question metrics.

    Note:
        judgment is None when the judge did not run (--no-judge, or the engine
        did not answer), so the claim metrics are left empty.
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
                      partial_claims=0, key_points_covered=None, judge_cost_usd=0.0)
        return record

    verdicts = [claim["verdict"] for claim in judgment["claims"]]
    covered = [point["covered"] for point in judgment["key_points"]]
    record.update(
        claims=judgment["claims"],
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
        # Only the agent reports the fields below. Workflow records don't
        # have them.
        "stop_reasons": dict(Counter(r.get("stop_reason") or "n/a" for r in records)),
        "routes": dict(Counter(r.get("route") or "n/a" for r in records)),
        "escalations": sum(1 for r in records if r.get("escalated")),
        "runs_with_parallel_calls": sum(1 for r in records if r.get("parallel_turns")),
        "avg_review_rounds": mean([r.get("review_rounds") for r in records if "review_rounds" in r]),
    }


def cost_from_usage(usage_by_model):
    """
    Compute the USD cost of one agent run from its per-model token counts.

    Returns None if any model used in the run has no price.
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
    """
    For each pair, compare the normal question with its variants.

    Why we need this:
    A "variant-only rejection" is the failure this evaluation is looking for.
    It happens when the engine answers the normal wording but rejects the
    typo, paraphrase, or abbreviation of the same question, which means the
    wording alone caused the rejection.
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
    """
    Run one engine over the question set and save the results.

    Args:
        args: Parsed command line arguments (--engine, --label, --ids,
            --repeat, --no-judge)
    """
    with open(QUESTIONS_FILE, encoding="utf-8") as handle:
        questions = json.load(handle)["questions"]
    if args.ids:
        wanted = set(args.ids.split(","))
        questions = [q for q in questions if q["id"] in wanted]

    engine = build_engine(args.engine)
    client = None if args.no_judge else OpenAI()

    # The judge only waits on the API, so we let it grade in background
    # threads while the next question runs. The engine itself still runs one
    # question at a time, so other questions don't affect its latency.
    pending = []  # (item, record, future or None), in question order
    with ThreadPoolExecutor(max_workers=JUDGE_THREADS) as judge_pool:
        for item in questions:
            for run_index in range(args.repeat):
                record = run_question(args.engine, engine, item)
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
        "engine": args.engine,
        "label": args.label,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "llm_model": config.LLM_MODEL,
            "judge_model": None if args.no_judge else JUDGE_MODEL,
            "judge_policy": None if args.no_judge else JUDGE_POLICY,
            "candidate_top_k": config.CANDIDATE_TOP_K,
            "final_top_n": config.FINAL_TOP_N,
            "min_doc_relevance": config.MIN_DOC_RELEVANCE,
            "off_topic_threshold": config.OFF_TOPIC_THRESHOLD,
            # Models, feature flags, and budgets the agent actually ran with.
            "agent": engine.features() if args.engine == "agent" else None,
        },
        "summary": summarize(records),
        "by_category": summarize_by_category(records),
        "pairs": pair_report(records),
        "judge_cost_usd": round(sum(r["judge_cost_usd"] for r in records), 4),
        "records": records,
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{args.engine}_{args.label + '_' if args.label else ''}{stamp}.json"
    raw_path = os.path.join(RAW_RESULTS_DIR, name)
    write_json(raw_path, output)
    path = os.path.join(RESULTS_DIR, name)
    write_json(path, public_copy(output))

    print_records(records)
    print_summary(f"{args.engine} overall", output["summary"])
    for category, summary in output["by_category"].items():
        print(f"  [{category}] acc={summary['mode_accuracy']} false_rej={summary['false_rejection_rate']} "
              f"unsup={summary['unsupported_claim_rate']} lat={summary['latency_mean_s']}")
    print("\nPairs:", json.dumps(output["pairs"], indent=1))
    print(f"\nJudge cost (not part of engine cost): ${output['judge_cost_usd']}")
    print(f"Saved {path} (public, no chunk text) and {raw_path} (local only)")


# ================================
# Public export
# ================================
def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, default=str)


def public_copy(output):
    """
    Return a copy of the results that is safe to publish.

    Why we need this:
    The source papers are not ours to publish, so we keep everything except
    their text. Each evidence entry still keeps its chunk id, title, URL, file
    name, and page, so every citation can still be traced back to its paper.
    """
    public = json.loads(json.dumps(output, default=str))
    for record in public.get("records", []):
        record["evidence"] = [
            {key: value for key, value in entry.items() if key != "text"}
            for entry in record.get("evidence", [])
        ]
    public["chunk_text_removed"] = True
    return public


def make_public(paths):
    """
    Back up result files to eval/results/raw/ and write their public copies.

    NOTE: This is safe to re-run. We skip a file that is already public, so it
    can never overwrite its raw backup, and we keep any raw backup that
    already exists instead of replacing it.
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


def rejudge(paths):
    """
    Re-run only the judge against saved engine results.

    Why we need this:
    Judge instructions can improve independently of the engine. The raw result
    already contains the exact answer and evidence from the original run, so we
    can apply the new grading rule without paying for or changing the engine
    run itself.

    Args:
        paths: Raw or public result JSON files. For a public file, the matching
            file in eval/results/raw/ supplies the evidence text.
    """
    with open(QUESTIONS_FILE, encoding="utf-8") as handle:
        questions = {item["id"]: item for item in json.load(handle)["questions"]}

    client = OpenAI()
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

        pending = []
        with ThreadPoolExecutor(max_workers=JUDGE_THREADS) as judge_pool:
            for record in output["records"]:
                item = questions[record["id"]]
                future = None
                if record["mode"] == "RAG" and record.get("answer"):
                    future = judge_pool.submit(judge_answer, client, item, record)
                pending.append((item, record, future))

            records = [
                add_scores(item, record, future.result() if future else None)
                for item, record, future in pending
            ]

        original_label = output.get("label") or "run"
        output["engine_created_at"] = output.get("engine_created_at", output.get("created_at"))
        output["created_at"] = datetime.now().isoformat(timespec="seconds")
        output["label"] = f"{original_label}_rejudged"
        output.setdefault("config", {})["judge_model"] = JUDGE_MODEL
        output["config"]["judge_policy"] = JUDGE_POLICY
        output["summary"] = summarize(records)
        output["by_category"] = summarize_by_category(records)
        output["pairs"] = pair_report(records)
        output["judge_cost_usd"] = round(sum(r["judge_cost_usd"] for r in records), 4)
        output["records"] = records

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{output['engine']}_{output['label']}_{stamp}.json"
        raw_path = os.path.join(RAW_RESULTS_DIR, name)
        public_path = os.path.join(RESULTS_DIR, name)
        write_json(raw_path, output)
        write_json(public_path, public_copy(output))
        print_summary(output["label"], output["summary"])
        print(f"Saved {public_path} (public, no chunk text) and {raw_path} (local only)")


def compare(paths, output_name="comparison.md"):
    """
    Print and save a side-by-side comparison of two or more result files.

    Args:
        paths: Result JSON files to compare (two or more)
        output_name: Markdown file name for the report, saved in eval/results
    """
    runs = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            runs.append(json.load(handle))
    names = [run["engine"] + (f" ({run['label']})" if run.get("label") else "") for run in runs]

    # Recompute the summaries from the records, so older result files also get
    # every metric and agent costs use the current price table.
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

    lines = table_header("Metric")
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

    # List the questions whose mode differs between runs. We only look at the
    # first run of each question.
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
        handle.write(f"# Engine comparison\n\n{files}\n\n{report}\n")
    print(f"\nSaved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", choices=["workflow", "agent"])
    parser.add_argument("--label", default="", help="Tag added to the results file name")
    parser.add_argument("--ids", default="", help="Comma-separated question ids to run")
    parser.add_argument("--repeat", type=int, default=1, help="Runs per question (stochastic checks)")
    parser.add_argument("--no-judge", action="store_true", help="Skip the claim-grounding judge")
    parser.add_argument("--compare", nargs="+", metavar="RESULT_JSON", help="Two or more result files")
    parser.add_argument("--out", default="comparison.md", help="File name for --compare output (in eval/results)")
    parser.add_argument("--make-public", nargs="+", metavar="RESULT_JSON",
                        help="Strip chunk text from existing result files (originals go to eval/results/raw/)")
    parser.add_argument("--rejudge", nargs="+", metavar="RESULT_JSON",
                        help="Re-run the judge on saved results without re-running the engine")
    args = parser.parse_args()

    if args.make_public:
        make_public(args.make_public)
    elif args.rejudge:
        rejudge(args.rejudge)
    elif args.compare:
        if len(args.compare) < 2:
            parser.error("--compare needs at least two result files.")
        compare(args.compare, args.out)
    elif args.engine:
        evaluate(args)
    else:
        parser.error("Use --engine, --compare, --make-public, or --rejudge.")


if __name__ == "__main__":
    main()
