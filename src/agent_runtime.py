"""Agent runtime for the OLED Assistant.

The agent plans the question, researches it in a tool-calling loop and answers within hard limits enforced in Python.

How one request flows:

1. PLAN: The planner labels the question `in_domain`, `out_of_domain` or `uncertain`, rates its complexity and creates sub-questions when needed. A clearly out-of-domain question stops as `OFF_TOPIC`.
2. ROUTE: Python selects the single model, the light or heavy model, or the orchestrator-worker team.
3. RESEARCH LOOP: The model uses `search_documents`, `submit_answer` and `declare_insufficient`. Search, filtering, reranking and citation validation run in Python. One turn can issue several searches.
4. ESCALATION: A recoverable light-model failure can restart once on the heavy model with the evidence already collected.

Every agent working on a question shares the same LLM-call, search, revision, review and time budgets. The deadline is checked around model calls and searches and before final acceptance. When a limit is reached, the run stops rather than returning an answer that could not be verified, and the trace records the reason.
"""

import json
import threading
import time

from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, OpenAI

import config
from agent_prompts import (
    PLAN_INSTRUCTIONS,
    PLAN_SCHEMA,
    TOOL_DECLARE,
    TOOL_SEARCH,
    TOOL_SUBMIT_ANSWER,
    research_instructions,
)
from agent_team import answer_with_team, review_answer
from agent_tools import (
    AgentStop,
    EvidenceLedger,
    SearchTool,
    format_evidence,
    normalize_query,
    number_citations,
    search_result_for_model,
    validate_submission,
)
from utils import estimate_cost_usd, logger

# ================================
# Stop reasons -> user-facing mode
# ================================
# The UI exposes four answer modes plus ERROR, while the trace keeps each detailed stop reason for evaluation.
STOP_REASON_TO_MODE = {
    "answered": "RAG",
    "out_of_domain": "OFF_TOPIC",
    "insufficient_evidence": "NO_ANSWER_IN_DOCS",
    "llm_budget_exhausted": "NO_ANSWER_IN_DOCS",
    "deadline_exceeded": "NO_ANSWER_IN_DOCS",
    "revision_limit": "NO_ANSWER_IN_DOCS",
    "review_failed": "NO_ANSWER_IN_DOCS",
    "api_timeout": "ERROR",
    "api_error": "ERROR",
    "clarification_needed": "CLARIFICATION",
}

STOP_MESSAGES = {
    "out_of_domain": "No Answer: The question is not related to OLED display technology.",
    "insufficient_evidence": "No Answer: I could not find enough evidence in the documents to answer this.",
    "llm_budget_exhausted": (
        "No Answer: I could not find verifiable evidence within the search budget."
    ),
    "deadline_exceeded": "No Answer: I could not verify an answer within the time limit.",
    "revision_limit": "No Answer: I could not produce an answer with valid citations.",
    "review_failed": (
        "No Answer: I could not verify an answer to the main question from the cited evidence."
    ),
    "api_timeout": "The language model timed out. Please try again.",
    "api_error": "The language model request failed. Please try again.",
    "clarification_needed": "Could you clarify what you would like to know?",
}


# ================================
# Per-request state
# ================================
class AgentRun:
    """Hold the state shared by every agent working on one question.

    Workers update counters and trace events under a lock because they run in separate threads.
    Nothing is shared between user requests, and every `AgentAssistant.query()` starts with a fresh `AgentRun`.
    """

    def __init__(self, question, on_event):
        self.question = question
        self.on_event = on_event
        self.started_at = time.monotonic()
        self.ledger = EvidenceLedger()
        self.search_cache = {}  # maps a normalized query to the chunk ids it returned
        self.plan = None
        self.route = None
        self.trace = []
        self.searches = 0
        self.llm_calls = 0
        self.revisions = 0
        self.review_rounds = 0
        self.invalid_turns = 0
        self.parallel_turns = 0  # number of turns where the model made 2+ tool calls
        self.escalated = False
        self.worker_reports = []
        self.max_relevance = 0.0
        self.usage_by_model = {}
        self.calls_by_role = {}
        self._lock = threading.RLock()
        # Only the thread that created the run may call `on_event`.
        # Streamlit rejects worker-thread callbacks, so worker events stay queued until the main thread flushes them.
        self._owner_thread = threading.get_ident()
        self._pending_events = []

    def snapshot(self):
        """JSON-safe request state for graph boundaries; never persist locks or clients.

        This snapshot resumes coarse nodes within one invocation.
        Public APIs start a fresh request instead of replaying a partly completed paid API call.
        """
        fields = (
            "question", "started_at", "search_cache", "plan", "route", "trace",
            "searches", "llm_calls", "revisions", "review_rounds", "invalid_turns",
            "parallel_turns", "escalated", "worker_reports", "max_relevance",
            "usage_by_model", "calls_by_role",
        )
        data = {key: getattr(self, key) for key in fields}
        data["ledger"] = [{k: v for k, v in e.items() if k != "doc"} for e in self.ledger.entries()]
        return json.loads(json.dumps(data))

    @classmethod
    def restore(cls, data, on_event=None):
        run = cls(data["question"], on_event)
        for key, value in data.items():
            if key != "ledger":
                setattr(run, key, value)
        for entry in data["ledger"]:
            run.ledger.add(entry["chunk_id"], entry)
        return run

    def elapsed(self):
        return time.monotonic() - self.started_at

    def time_left(self):
        return config.AGENT_DEADLINE_SECONDS - self.elapsed()

    def check_deadline(self, stage):
        """Stop the run if the wall-clock deadline has passed or is less than a second away."""
        if self.time_left() <= 1:
            raise AgentStop("deadline_exceeded", f"{stage}: {self.elapsed():.1f}s elapsed")

    # ---------- budgets ----------
    def reserve_llm_call(self, role):
        """Count one LLM call against the budget shared by every agent on this question."""
        with self._lock:
            if self.llm_calls >= config.AGENT_MAX_LLM_CALLS:
                raise AgentStop("llm_budget_exhausted", f"{self.llm_calls} LLM calls used")
            self.check_deadline(f"{role} call")
            self.llm_calls += 1
            self.calls_by_role[role] = self.calls_by_role.get(role, 0) + 1

    def llm_calls_left(self):
        return config.AGENT_MAX_LLM_CALLS - self.llm_calls

    def reserve_search(self):
        """Count one search against the shared budget. Returns False when none are left."""
        with self._lock:
            if self.searches >= config.AGENT_MAX_SEARCHES:
                return False
            self.searches += 1
            return True

    def searches_left(self):
        return config.AGENT_MAX_SEARCHES - self.searches

    def add_usage(self, model, usage):
        """Add up the Responses API token usage for each model."""
        if usage is None:
            return
        input_details = getattr(usage, "input_tokens_details", None)
        output_details = getattr(usage, "output_tokens_details", None)
        with self._lock:
            totals = self.usage_by_model.setdefault(
                model,
                {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0},
            )
            totals["calls"] += 1
            totals["input_tokens"] += usage.input_tokens or 0
            totals["output_tokens"] += usage.output_tokens or 0
            totals["cached_tokens"] += getattr(input_details, "cached_tokens", 0) or 0
            totals["reasoning_tokens"] += getattr(output_details, "reasoning_tokens", 0) or 0

    def usage_summary(self):
        """Return the totals across all models, plus the cost (None if any model has no price)."""
        totals = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0}
        cost = 0.0
        for model, usage in self.usage_by_model.items():
            for key in totals:
                totals[key] += usage[key]
            model_cost = estimate_cost_usd(
                model, usage["input_tokens"], usage["output_tokens"], usage["cached_tokens"]
            )
            cost = None if (cost is None or model_cost is None) else cost + model_cost
        return {
            "llm_calls": self.llm_calls,
            "prompt_tokens": totals["input_tokens"],
            "completion_tokens": totals["output_tokens"],
            "cached_tokens": totals["cached_tokens"],
            "reasoning_tokens": totals["reasoning_tokens"],
            "cost_usd": cost,
            "by_model": self.usage_by_model,
            "calls_by_role": self.calls_by_role,
        }

    # ---------- trace ----------
    def record(self, event_type, role=None, **fields):
        """Append a trace event and forward it to the UI callback.

        Worker-thread events are queued here and sent later by the main thread through `flush_events()`.
        """
        with self._lock:
            event = {
                "step": len(self.trace) + 1,
                "type": event_type,
                "role": role,
                "t": round(self.elapsed(), 2),
                **fields,
            }
            self.trace.append(event)
            self._pending_events.append(event)
        if threading.get_ident() == self._owner_thread:
            self.flush_events()
        return event

    def flush_events(self):
        """Send queued events to the UI callback (main thread only)."""
        with self._lock:
            events, self._pending_events = self._pending_events, []
        if self.on_event is None:
            return
        for event in events:
            try:
                self.on_event(event)
            except Exception as exc:  # noqa: BLE001 - a broken UI callback should never end the run
                logger.warning("on_event callback failed: %s", exc)


# ================================
# Agent
# ================================
class AgentAssistant:
    """Plan, route, research with tools, review and answer OLED questions.

    One shared `Retriever` keeps the embedding model, vector store and reranker loaded once.
    """

    def __init__(self, retriever, client=None):
        self.retriever = retriever
        self.search_tool = SearchTool(retriever)
        # Disable SDK retries so `respond()` can keep every retry inside the request deadline.
        self.client = client if client is not None else OpenAI(timeout=config.AGENT_API_TIMEOUT_SECONDS, max_retries=0)
        self.parallel = config.AGENT_PARALLEL_SEARCH
        if config.AGENT_ROUTING:
            self.light_model = config.AGENT_LIGHT_MODEL
            self.heavy_model = config.AGENT_HEAVY_MODEL
        else:
            # Without routing, every role uses the same model.
            self.light_model = self.heavy_model = config.AGENT_MODEL

    def features(self):
        """Return the active configuration. We store it with every result so evaluation knows what was on."""
        return {
            "parallel_search": self.parallel,
            "routing": config.AGENT_ROUTING,
            "workers": config.AGENT_WORKERS,
            "reviewer": config.AGENT_REVIEWER,
            "light_model": self.light_model,
            "heavy_model": self.heavy_model,
            "max_llm_calls": config.AGENT_MAX_LLM_CALLS,
            "max_searches": config.AGENT_MAX_SEARCHES,
        }

    # ---------- one LLM call: budget, deadline, usage ----------
    def respond(self, run, role, model, input_items, tools=None, text_format=None):
        """Make one Responses API call for `role`.

        Every agent LLM call passes through this method for budget accounting, deadline checks, transient-error retries and token usage.
        `run` is the current `AgentRun`, `model` is the OpenAI model name, and `input_items` becomes the request input.
        Optional `tools` require a tool call, while `text_format` defines structured output.
        The method returns the Responses API response object.
        `store=False` prevents OpenAI response storage, and encrypted reasoning items are passed back explicitly on the next turn.
        Budget, deadline and API failures raise `AgentStop`.
        """
        run.reserve_llm_call(role)
        kwargs = {"model": model, "input": input_items, "store": False}
        effort = config.reasoning_effort_for(model)
        if effort:
            kwargs["reasoning"] = {"effort": effort}
            kwargs["include"] = ["reasoning.encrypted_content"]
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "required"
            kwargs["parallel_tool_calls"] = self.parallel
        if text_format:
            kwargs["text"] = {"format": text_format}

        # Every retry and its pause must fit inside the remaining request time.
        retry_pause = 1.0
        for attempt in range(config.AGENT_API_RETRIES + 1):
            run.check_deadline(f"{role} call")
            kwargs["timeout"] = min(config.AGENT_API_TIMEOUT_SECONDS, run.time_left())
            try:
                response = self.client.responses.create(**kwargs)
                break
            except APITimeoutError as exc:
                # A timeout caused by the remaining-time cap is reported as a deadline stop.
                reason = "deadline_exceeded" if run.time_left() <= 1 else "api_timeout"
                stop = AgentStop(reason, f"{role}: {exc}")
            except APIConnectionError as exc:
                stop = AgentStop("api_error", f"{role}: {exc}")
            except APIStatusError as exc:
                stop = AgentStop("api_error", f"{role}: {exc}")
                if exc.status_code != 429 and exc.status_code < 500:
                    raise stop from exc  # e.g. bad request or auth error, a retry would fail the same way
            except APIError as exc:
                raise AgentStop("api_error", f"{role}: {exc}") from exc

            if attempt == config.AGENT_API_RETRIES or run.time_left() <= retry_pause + 2:
                raise stop
            run.record("api_retry", role=role, reason=stop.reason, detail=stop.detail[:200])
            time.sleep(retry_pause)

        run.add_usage(model, response.usage)
        return response

    # ---------- Step 1: plan ----------
    def _plan(self, run):
        response = self.respond(
            run,
            "planner",
            self.light_model,
            [
                {"role": "developer", "content": PLAN_INSTRUCTIONS},
                {"role": "user", "content": run.question},
            ],
            text_format={"type": "json_schema", "name": "plan", "schema": PLAN_SCHEMA, "strict": True},
        )
        try:
            plan = json.loads(response.output_text or "")
        except json.JSONDecodeError:
            # A malformed plan should not block the question.
            # Fall back to an `"uncertain"` plan and search the original question.
            plan = {"domain": "uncertain", "reason": "plan output unreadable"}

        subquestions = [s.strip() for s in plan.get("subquestions", []) if s and s.strip()]
        plan["subquestions"] = subquestions[: config.AGENT_MAX_SUBQUESTIONS] or [run.question]
        plan["complexity"] = plan.get("complexity", "simple")
        plan["sequential"] = bool(plan.get("sequential", False))
        run.plan = plan
        run.record(
            "plan",
            role="planner",
            model=self.light_model,
            domain=plan["domain"],
            reason=plan["reason"],
            complexity=plan["complexity"],
            sequential=plan["sequential"],
            subquestions=plan["subquestions"],
        )
        return plan

    # ---------- search tool handler (shared by every role) ----------
    def handle_search(self, run, role, args, seen_chunk_ids):
        """Run one `search_documents` call for `role`.

        `seen_chunk_ids` tracks chunks already sent in full to this model context so their text is not repeated.
        """
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return {"status": "error", "message": "query must be a non-empty string."}
        query = query.strip()
        if len(query) > config.AGENT_MAX_QUERY_CHARS:
            return {"status": "error", "message": f"query is longer than {config.AGENT_MAX_QUERY_CHARS} characters."}

        # Repeated queries use the request cache without spending another search.
        # A different agent context still receives any chunk text it has not seen.
        key = normalize_query(query)
        cached = run.search_cache.get(key)
        if cached is not None:
            unseen = [chunk_id for chunk_id in cached["chunk_ids"] if chunk_id not in seen_chunk_ids]
            run.record("search_repeat", role=role, query=query, reused_chunks=len(unseen))
            if not unseen:
                return {
                    "status": "already_searched",
                    "message": "You already have these results. Use them or try a different query.",
                    "chunk_ids": cached["chunk_ids"],
                }
            response = {**cached, "results": [run.ledger.get(chunk_id) for chunk_id in cached["chunk_ids"]]}
            result = search_result_for_model(response, seen_chunk_ids)
            result["note"] = "Same query was already run in this request; results reused."
            result["searches_left"] = run.searches_left()
            return result
        run.check_deadline(f"{role} before search")
        if not run.reserve_search():
            return {
                "status": "search_budget_exhausted",
                "message": "No searches left. Finish with the evidence you have.",
            }

        response = self.search_tool.search(query, run.ledger)
        # Searches can wait on the shared CPU lock, so check the deadline again afterward.
        run.check_deadline(f"{role} after search")
        with run._lock:
            run.max_relevance = max(run.max_relevance, response["max_relevance"])
            run.search_cache[key] = {
                "status": response["status"],
                "max_relevance": response["max_relevance"],
                "min_doc_relevance": response["min_doc_relevance"],
                "chunk_ids": [item["chunk_id"] for item in response["results"]],
            }
        run.record(
            "search",
            role=role,
            query=query,
            status=response["status"],
            max_relevance=response["max_relevance"],
            survivor_count=response["survivor_count"],
            result_count=len(response["results"]),
            new_chunks=sum(1 for item in response["results"] if item["is_new"]),
        )
        result = search_result_for_model(response, seen_chunk_ids)
        result["searches_left"] = run.searches_left()
        return result

    # ---------- Step 3: generic research loop ----------
    def research(self, run, role, model, instructions, user_content, finish_tools, handle_finish, given_chunk_ids=()):
        """Run the tool loop shared by single agents, workers, orchestrators and escalated heavy agents.

        Each turn calls the model with its currently available tools, executes the returned tool calls and sends their results back.
        The loop ends when `handle_finish` returns `("done", value)` or raises `AgentStop`.
        `role` identifies the caller for budgets and trace events.
        `model`, `instructions` and `user_content` define its initial context.
        `finish_tools` lists the tools that may end the loop.
        `given_chunk_ids` identifies evidence already included in `user_content`.
        Without given evidence, the loop must search before it can finish.
        The return value is the payload returned with `"done"`.
        """
        input_items = [
            {"role": "developer", "content": instructions},
            {"role": "user", "content": user_content},
        ]
        finish_names = {tool["name"] for tool in finish_tools}
        # This context may cite only chunks whose full text it has received.
        seen_chunk_ids = set(given_chunk_ids)
        require_search = not seen_chunk_ids
        calls_used = 0
        searched_here = False

        while True:
            if calls_used >= config.AGENT_MAX_CALLS_PER_LOOP:
                raise AgentStop("llm_budget_exhausted", f"{role} used its {calls_used} calls")

            # Remove search on the loop's final call so the model must finish with existing evidence.
            # The final submission is still validated normally.
            last_call = calls_used >= config.AGENT_MAX_CALLS_PER_LOOP - 1 or run.llm_calls_left() <= 1
            tools = []
            if run.searches_left() > 0 and not last_call:
                tools.append(TOOL_SEARCH)
            if searched_here or not require_search:
                tools.extend(finish_tools)
            if not tools:
                raise AgentStop("llm_budget_exhausted", f"{role}: no tool available")

            response = self.respond(run, role, model, input_items, tools=tools)
            calls_used += 1
            # Keep model output and encrypted reasoning so the next turn continues the same context.
            input_items.extend(item.model_dump(exclude_none=True) for item in response.output)
            calls = [item for item in response.output if item.type == "function_call"]

            # Plain-text output is an invalid turn because this loop requires tool calls.
            if not calls:
                with run._lock:
                    run.invalid_turns += 1
                run.record("no_tool_call", role=role)
                input_items.append({"role": "user", "content": "Respond by calling one of the provided tools."})
                continue

            if len(calls) > 1:
                with run._lock:
                    run.parallel_turns += 1
                run.record("parallel_calls", role=role, tools=[call.name for call in calls])

            allowed = {tool["name"] for tool in tools}
            for call in calls:
                result = None
                try:
                    args = json.loads(call.arguments or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("arguments must be a JSON object")
                except (json.JSONDecodeError, ValueError) as exc:
                    with run._lock:
                        run.invalid_turns += 1
                    run.record("bad_arguments", role=role, tool=call.name, error=str(exc))
                    result = {"status": "error", "message": f"Arguments are not valid JSON: {exc}"}
                else:
                    if call.name not in allowed:
                        with run._lock:
                            run.invalid_turns += 1
                        run.record("unknown_tool", role=role, tool=call.name)
                        result = {"status": "error", "message": f"Tool '{call.name}' is not available now. Use one of: {sorted(allowed)}."}
                    elif call.name == "search_documents":
                        result = self.handle_search(run, role, args, seen_chunk_ids)
                        searched_here = True
                    elif call.name in finish_names and len(calls) > 1:
                        # A finish call cannot share a turn with search because the model has not read the new results.
                        result = {"status": "error", "message": f"{call.name} must be called alone, after reading the search results."}
                    else:
                        outcome, payload = handle_finish(call.name, args)
                        if outcome == "done":
                            return payload
                        result = payload
                        if result.get("status") == "rejected":
                            # Return valid IDs so a miscopied chunk hash can be corrected in one turn.
                            result["citable_chunk_ids"] = sorted(seen_chunk_ids)
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(result, ensure_ascii=False),
                    }
                )

    # ---------- finish handler for anyone writing the final answer ----------
    def answer_handler(self, run, role):
        """Build the `submit_answer` and `declare_insufficient` handlers for `role`.

        Citation checks run before optional claim-level review.
        `declare_insufficient` ends the run with `insufficient_evidence`.
        """
        state = {"revisions": 0}

        def handle(name, args):
            if name == "declare_insufficient":
                missing = str(args.get("missing", "")).strip()[: config.AGENT_MAX_QUERY_CHARS]
                raise AgentStop("insufficient_evidence", missing)

            answer = args.get("answer")
            citations = args.get("citations")
            # A citation rejection returns to the writer until its revision budget is exhausted.
            errors = validate_submission(answer, citations, run.ledger, config.AGENT_MAX_ANSWER_CHARS)
            if errors:
                state["revisions"] += 1
                with run._lock:
                    run.revisions += 1
                run.record("answer_rejected", role=role, errors=errors, revision=state["revisions"])
                if state["revisions"] > config.AGENT_MAX_ANSWER_REVISIONS:
                    raise AgentStop("revision_limit", "; ".join(errors))
                return "continue", {
                    "status": "rejected",
                    "errors": errors,
                    "revisions_left": config.AGENT_MAX_ANSWER_REVISIONS - state["revisions"] + 1,
                }

            citations = citations if isinstance(citations, list) else []
            # 2) Optional claim-by-claim review of a citation-valid answer.
            if config.AGENT_REVIEWER:
                feedback = review_answer(self, run, answer, citations)
                if feedback is not None:
                    return "continue", feedback
            # The deadline covers the whole request, including final acceptance.
            run.check_deadline(f"{role} final answer")
            return "done", (answer, citations)

        return handle

    # ---------- single agent (light, heavy, or the only model) ----------
    def answer_alone(self, run, role, model, prior_evidence=None):
        """Let one agent research and answer the complete question.

        On escalation, `prior_evidence` contains original chunks already collected in this request.
        Supplying that text allows the new loop to finish without repeating a search.
        """
        plan_lines = "\n".join(f"{i}. {s}" for i, s in enumerate(run.plan["subquestions"], 1))
        content = (
            f"Question: {run.question}\n\n"
            f"Plan (cover each item):\n{plan_lines}\n\n"
            f"Searches left in this request: {run.searches_left()}."
        )
        if prior_evidence:
            content += (
                "\n\nEvidence already retrieved in this request (cite these ids, "
                "or search for more):\n\n" + format_evidence(prior_evidence)
            )
        return self.research(
            run,
            role,
            model,
            research_instructions(self.parallel),
            content,
            [TOOL_SUBMIT_ANSWER, TOOL_DECLARE],
            self.answer_handler(run, role),
            given_chunk_ids=[entry["chunk_id"] for entry in prior_evidence or []],
        )

    # ---------- Step 2: routing ----------
    def _choose_route(self, plan):
        if config.AGENT_WORKERS >= 2 and len(plan["subquestions"]) >= 2:
            return "team"
        if not config.AGENT_ROUTING:
            return "single"
        return "heavy" if plan["complexity"] == "complex" else "light"

    def _should_escalate(self, run, stop):
        """Decide whether a failed light run should retry once on the heavy model.

        Escalation is limited to failures a stronger model could plausibly fix: citation revisions, exhausted loop turns, reviewer failure, or insufficient evidence despite strong retrieval.
        Out-of-domain questions, API failures and the global deadline never escalate.
        """
        if not config.AGENT_ROUTING or run.escalated or run.route != "light":
            return False
        if run.llm_calls_left() < 2 or run.time_left() < 20:
            return False
        if stop.reason in ("revision_limit", "llm_budget_exhausted", "review_failed"):
            return True
        return (
            stop.reason == "insufficient_evidence"
            and run.max_relevance >= config.AGENT_ESCALATE_MIN_RELEVANCE
        )

    def _answer(self, run):
        """Route the question and return (answer, citations), or raise AgentStop."""
        route = self._choose_route(run.plan)
        run.route = route
        run.record("route", role="planner", route=route)

        if route == "team":
            return answer_with_team(self, run)

        role = {"single": "agent", "light": "light", "heavy": "heavy"}[route]
        model = self.heavy_model if route == "heavy" else self.light_model
        try:
            return self.answer_alone(run, role, model)
        except AgentStop as stop:
            if not self._should_escalate(run, stop):
                raise
            run.escalated = True
            run.record(
                "escalate",
                role="heavy",
                from_model=self.light_model,
                to_model=self.heavy_model,
                reason=stop.reason,
                detail=stop.detail[:200],
            )
            # The heavy model receives the question, plan and original chunks in a fresh context without the light model's conversation.
            return self.answer_alone(run, "heavy", self.heavy_model, prior_evidence=run.ledger.entries())

    # ---------- Public entry point ----------
    def query(self, question, on_event=None):
        """Answer one question and return the answer, mode, retrieval metadata, sources and agent trace.

        Optional `on_event` receives live trace events on the main thread for interfaces such as Streamlit.
        """
        run = AgentRun(question, on_event)
        answer, citations = None, []
        stop_reason, detail = "answered", ""

        try:
            plan = self._plan(run)
            if plan["domain"] == "out_of_domain":
                raise AgentStop("out_of_domain", plan["reason"])
            answer, citations = self._answer(run)
        except AgentStop as stop:
            stop_reason, detail = stop.reason, stop.detail
        except Exception as exc:  # noqa: BLE001 - any other bug is reported as ERROR so the app keeps running
            logger.exception("Agent run failed")
            stop_reason, detail = "api_error", f"{type(exc).__name__}: {exc}"

        return self.result_from_run(run, answer, citations, stop_reason, detail)

    def result_from_run(self, run, answer=None, citations=None, stop_reason="answered", detail=""):
        """Shared response formatting for the legacy baseline and graph runtime."""
        citations = citations or []
        run.record("stop", reason=stop_reason, detail=detail[:300])
        run.flush_events()
        mode = STOP_REASON_TO_MODE[stop_reason]

        cited_entries = []
        if stop_reason == "answered":
            display_answer, cited_entries = number_citations(answer, citations, run.ledger)
        else:
            display_answer = STOP_MESSAGES[stop_reason]
            if stop_reason == "insufficient_evidence" and detail:
                display_answer += f"\n\nMissing: {detail}"

        # Keep all retrieved chunks for private failure analysis.
        # `evidence` remains limited to final citations, and public exports remove text from both fields.
        retrieved_entries = [
            {key: value for key, value in entry.items() if key != "doc"}
            for entry in run.ledger.entries()
        ]

        sources = [
            {
                "number": number,
                "title": entry["title"],
                "url": entry["url"],
                "file_name": entry["file_name"],
                "page": entry["page"],
                "chunk_id": entry["chunk_id"],
            }
            for number, entry in enumerate(cited_entries, 1)
        ]
        usage = run.usage_summary()
        logger.info(
            "🤖 Agent route=%s stop=%s mode=%s searches=%d llm_calls=%d %.1fs",
            run.route, stop_reason, mode, run.searches, run.llm_calls, run.elapsed(),
        )

        return {
            "answer": display_answer,
            "mode": mode,
            "relevance_score": run.max_relevance,
            "retrieval_metadata": {
                "engine": "agent",
                "route": run.route,
                "max_relevance": round(run.max_relevance, 3),
                "min_doc_relevance": self.retriever.min_doc_relevance,
                "searches": run.searches,
                "llm_calls": run.llm_calls,
                "ledger_size": len(run.ledger),
                "cited_chunks": len(cited_entries),
                "stop_reason": stop_reason,
            },
            "sources": sources,
            "agent": {
                "question": run.question,
                "features": self.features(),
                "plan": run.plan,
                "route": run.route,
                "escalated": run.escalated,
                "stop_reason": stop_reason,
                "stop_detail": detail,
                "searches": run.searches,
                "revisions": run.revisions,
                "review_rounds": run.review_rounds,
                "invalid_turns": run.invalid_turns,
                "parallel_turns": run.parallel_turns,
                "workers": run.worker_reports,
                "elapsed_s": round(run.elapsed(), 2),
                "usage": usage,
                "trace": run.trace,
                "evidence": [
                    {key: entry[key] for key in ("chunk_id", "title", "url", "file_name", "page", "text")}
                    for entry in cited_entries
                ],
                "retrieved_evidence": retrieved_entries,
            },
        }


def save_trace(result, path=config.AGENT_TRACE_PATH):
    """Append one agent result (without the document objects) to a JSONL trace file."""
    agent = result.get("agent")
    if not agent:
        return
    record = {**agent, "mode": result["mode"], "answer": result["answer"], "sources": result.get("sources", [])}
    try:
        with open(path, "a", encoding="utf-8") as trace_file:
            trace_file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError as exc:
        logger.warning("Could not write agent trace: %s", exc)
