"""Multi-agent patterns built on top of the shared OLED agent runtime.

Orchestrator-worker (`AGENT_WORKERS >= 2`) is used when the plan has at least two sub-questions.

- Each light-model worker has its own context and tool loop, while all workers share the request's search budget, cache and evidence ledger.
- Independent sub-questions run in parallel. Dependent sub-questions run in order, and each worker sees the earlier findings.
- Workers return findings and citations. The heavy-model orchestrator checks them against the original cited chunks and writes the final answer.

Reviewer (`AGENT_REVIEWER`) checks the question, answer and cited chunk text without seeing the research trace.

- An answer passes only when the review is readable, the core question is answered and no unsupported claim remains.
- Otherwise feedback returns to the same writer for deletion or further research, up to `AGENT_MAX_REVIEW_ROUNDS`.
- If the final review still fails, the run returns `NO_ANSWER_IN_DOCS` rather than a partial answer.

Every function receives the shared `AgentAssistant` and per-request `AgentRun`, so budgets, usage and trace data stay in one place.
"""

import json
from concurrent.futures import ThreadPoolExecutor

import config
from agent_prompts import (
    ORCHESTRATOR_INSTRUCTIONS,
    REVIEW_SCHEMA,
    REVIEWER_INSTRUCTIONS,
    TOOL_DECLARE,
    TOOL_SUBMIT_ANSWER,
    TOOL_SUBMIT_FINDINGS,
    worker_instructions,
)
from agent_tools import AgentStop, format_evidence, inline_citation_ids, validate_findings

# Run-level stops end the whole request even when they occur inside a worker.
_FATAL_STOPS = ("deadline_exceeded", "api_timeout", "api_error")


# ================================
# Orchestrator-worker
# ================================
def split_subquestions(subquestions, worker_count):
    """Split sub-questions into one contiguous group per worker.

    Contiguous groups preserve dependent-hop order.
    For example, three sub-questions on two workers become `[[q1, q2], [q3]]`.
    """
    worker_count = max(1, min(worker_count, len(subquestions)))
    size, extra = divmod(len(subquestions), worker_count)
    groups, start = [], 0
    for index in range(worker_count):
        end = start + size + (1 if index < extra else 0)
        groups.append(subquestions[start:end])
        start = end
    return groups


def _findings_handler(run, role):
    """Build the submit_findings / declare_insufficient handler for one worker."""
    state = {"revisions": 0}

    def handle(name, args):
        if name == "declare_insufficient":
            return "done", {"status": "insufficient", "missing": str(args.get("missing", ""))[:300]}

        errors = validate_findings(args.get("findings"), args.get("citations"), run.ledger)
        if errors:
            state["revisions"] += 1
            with run._lock:
                run.revisions += 1
            run.record("findings_rejected", role=role, errors=errors)
            if state["revisions"] > config.AGENT_MAX_ANSWER_REVISIONS:
                return "done", {"status": "failed", "missing": "; ".join(errors)}
            return "continue", {"status": "rejected", "errors": errors}

        # Keep only IDs that exist in the ledger after validation rejects unknown values.
        citations = [c for c in args["citations"] if c in run.ledger]
        return "done", {"status": "found", "findings": args["findings"].strip(), "citations": citations}

    return handle


def _run_worker(agent, run, number, subquestions, earlier_reports):
    """Research one group of sub-questions in a fresh context and return a report."""
    role = f"worker{number}"
    model = agent.light_model
    run.record("worker_start", role=role, model=model, subquestions=subquestions)

    content = (
        f"Overall question (for context only): {run.question}\n\n"
        "Your sub-question(s):\n" + "\n".join(f"- {s}" for s in subquestions)
    )
    # A sequential worker receives earlier findings so it knows what to search next.
    if earlier_reports:
        content += "\n\nFindings from earlier steps:\n" + "\n".join(
            f"- {r.get('findings') or r.get('missing', '')}" for r in earlier_reports
        )

    try:
        result = agent.research(
            run,
            role,
            model,
            worker_instructions(agent.parallel),
            content,
            [TOOL_SUBMIT_FINDINGS, TOOL_DECLARE],
            _findings_handler(run, role),
        )
    except AgentStop as stop:
        if stop.reason in _FATAL_STOPS:
            raise
        # One worker exhausting its loop does not stop the full request.
        # The orchestrator continues with the remaining reports.
        result = {"status": "failed", "missing": f"{stop.reason}: {stop.detail}"[:300]}

    report = {"worker": number, "model": model, "subquestions": subquestions, **result}
    run.record(
        "worker_done",
        role=role,
        status=report["status"],
        citations=len(report.get("citations", [])),
    )
    return report


def answer_with_team(agent, run):
    """Answer with research workers followed by a heavy-model orchestrator.

    Workers first research their assigned sub-question groups.
    The orchestrator then checks their findings against the cited chunks and returns the accepted answer and citations.
    `agent` supplies models, the API client and research loop, while `run` contains current request state.
    """
    groups = split_subquestions(run.plan["subquestions"], config.AGENT_WORKERS)

    reports = []
    if run.plan["sequential"]:
        # Dependent hops run in order, and each worker receives earlier findings.
        for number, group in enumerate(groups, 1):
            reports.append(_run_worker(agent, run, number, group, list(reports)))
    else:
        # Independent workers overlap LLM calls.
        # Their local CPU searches still run one at a time through `SearchTool`.
        with ThreadPoolExecutor(max_workers=len(groups)) as pool:
            futures = [
                pool.submit(_run_worker, agent, run, number, group, [])
                for number, group in enumerate(groups, 1)
            ]
            reports = [future.result() for future in futures]
        # Flush queued worker events from the main thread because workers cannot update Streamlit directly.
        run.flush_events()
    run.worker_reports = reports

    return _orchestrate(agent, run, reports)


def _orchestrate(agent, run, reports):
    """Let the heavy model check the worker findings against the chunk text and write the answer."""
    role = "orchestrator"

    # Collect each cited chunk once, in worker order.
    cited_ids = []
    for report in reports:
        for chunk_id in report.get("citations", []):
            if chunk_id not in cited_ids:
                cited_ids.append(chunk_id)
    cited_entries = [run.ledger.get(chunk_id) for chunk_id in cited_ids]

    worker_blocks = []
    for report in reports:
        block = (
            f"Worker {report['worker']}: " + " / ".join(report["subquestions"]) + "\n"
            f"Status: {report['status']}\n"
        )
        if report["status"] == "found":
            block += f"Findings: {report['findings']}"
        else:
            block += f"Missing: {report.get('missing', '')}"
        worker_blocks.append(block)

    content = (
        f"Question: {run.question}\n\n"
        "Worker reports:\n\n" + "\n\n".join(worker_blocks) + "\n\n"
        "Original text of the cited chunks:\n\n" + format_evidence(cited_entries) + "\n\n"
        f"Searches left in this request: {run.searches_left()}."
    )
    run.record("orchestrate", role=role, model=agent.heavy_model, chunks=len(cited_entries))

    return agent.research(
        run,
        role,
        agent.heavy_model,
        ORCHESTRATOR_INSTRUCTIONS,
        content,
        [TOOL_SUBMIT_ANSWER, TOOL_DECLARE],
        agent.answer_handler(run, role),
        given_chunk_ids=cited_ids,
    )


# ================================
# Reviewer
# ================================
def review_answer(agent, run, answer, citations):
    """Check every claim in a citation-valid answer against its cited chunks.

    `validate_submission` proves only that citations reference evidence shown to the model.
    The reviewer determines whether that evidence supports each claim.
    Inline citation IDs are combined with the submitted citation list.
    The function returns `None` to accept the answer or structured feedback for the same writer context.
    An unreadable final review, unsupported claim, or unanswered core question raises `AgentStop("review_failed")`.
    The system therefore fails closed instead of returning an unverified answer.
    """
    role = "reviewer"
    cited_ids = []
    for chunk_id in list(citations) + inline_citation_ids(answer):
        if chunk_id in run.ledger and chunk_id not in cited_ids:
            cited_ids.append(chunk_id)
    entries = [run.ledger.get(chunk_id) for chunk_id in cited_ids]

    response = agent.respond(
        run,
        role,
        agent.heavy_model,
        [
            {"role": "developer", "content": REVIEWER_INSTRUCTIONS},
            {
                "role": "user",
                "content": (
                    f"Question: {run.question}\n\n"
                    f"Answer:\n{answer}\n\n"
                    f"Cited chunks:\n\n{format_evidence(entries)}"
                ),
            },
        ],
        text_format={"type": "json_schema", "name": "review", "schema": REVIEW_SCHEMA, "strict": True},
    )
    with run._lock:
        run.review_rounds += 1
        review_round = run.review_rounds

    review = _parse_review(response.output_text)
    if review is None:
        # An unreadable review cannot approve an answer.
        run.record("review_unreadable", role=role, round=review_round)
        raise AgentStop("review_failed", f"reviewer output unreadable (round {review_round})")

    unsupported = [claim for claim in review["claims"] if not claim["supported"]]
    core_answered = review["core_question_answered"]
    run.record(
        "review",
        role=role,
        model=agent.heavy_model,
        round=review_round,
        claims=len(review["claims"]),
        unsupported=len(unsupported),
        core_question_answered=core_answered,
        unsupported_claims=[claim["claim"][:160] for claim in unsupported],
    )
    # Approval requires an answered core question and support for every claim.
    # Revised answers return through `submit_answer` and are reviewed again.
    if core_answered and not unsupported:
        return None

    if review_round > config.AGENT_MAX_REVIEW_ROUNDS:
        problems = []
        if unsupported:
            problems.append(f"{len(unsupported)} unsupported claim(s)")
        if not core_answered:
            problems.append("core question not answered")
        raise AgentStop("review_failed", f"{' and '.join(problems)} after {review_round} reviews")

    instructions = []
    if unsupported:
        instructions.append(
            "Fix every unsupported claim: 'delete' means remove the claim; "
            "'research' means search for evidence, and remove the claim if none is found."
        )
    if not core_answered:
        instructions.append(
            "The answer does not give what the question asks for. Search for "
            "the missing core information."
        )
    return {
        "status": "review_failed",
        "message": (
            "A reviewer checked your answer against the cited chunks. "
            + " ".join(instructions)
            + " Then call submit_answer again. If the evidence cannot answer "
            "the core question, call declare_insufficient."
        ),
        "core_question_answered": core_answered,
        "unsupported_claims": [
            {"claim": claim["claim"], "fix": claim["fix"], "reason": claim["reason"]}
            for claim in unsupported
        ],
        "searches_left": run.searches_left(),
        "review_rounds_left": config.AGENT_MAX_REVIEW_ROUNDS - review_round + 1,
    }


def _parse_review(output_text):
    """Parse the reviewer JSON. Returns None if it is missing, malformed, or incomplete."""
    try:
        review = json.loads(output_text or "")
    except json.JSONDecodeError:
        return None
    if not isinstance(review, dict) or not isinstance(review.get("core_question_answered"), bool):
        return None
    # A cited answer must contain at least one claim.
    # An empty list means the reviewer checked nothing and cannot approve the answer.
    claims = review.get("claims")
    if not isinstance(claims, list) or not claims:
        return None
    for claim in claims:
        if not isinstance(claim, dict) or not isinstance(claim.get("supported"), bool):
            return None
        if not isinstance(claim.get("claim"), str) or not claim["claim"].strip():
            return None
        claim.setdefault("fix", "none")
        claim.setdefault("reason", "")
    return review
