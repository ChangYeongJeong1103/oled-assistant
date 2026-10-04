"""
Agent Runtime for OLED Assistant

The agent plans the question, researches it in a tool-calling loop, and then
answers. Every step runs inside hard limits that we enforce in Python.

How one request flows:

  1. PLAN (1 LLM call, structured output)
     The planner labels the question in_domain / out_of_domain / uncertain,
     rates its complexity, and splits it into sub-questions only when it
     needs to. A clearly out-of-domain question stops here as OFF_TOPIC.

  2. ROUTE (no LLM call)
     - single model:  AGENT_MODEL does everything
     - routing on:    simple questions go to the light model, complex ones
                      to the heavy model
     - workers on:    2+ sub-questions go to orchestrator-worker (agent_team.py)

  3. RESEARCH LOOP (Responses API, tool calling)
     search_documents(query)        retrieve, filter, and rerank in Python
     submit_answer(answer, cites)   citations are checked against the ledger,
                                    then optionally reviewed claim by claim
     declare_insufficient(missing)  stop with NO_ANSWER_IN_DOCS
     With parallel search on, one turn may issue several searches.

  4. ESCALATION (routing only, at most once)
     If the light model fails in a way we recognise, the heavy model starts
     over in a fresh context with the evidence we already collected.

  We keep every limit in Python so the model can't ignore it: total LLM
  calls, calls per loop, searches, answer revisions, review rounds, a
  wall-clock deadline, and per-call API timeouts. Every agent working on the
  same question shares these limits. We check the deadline before every API
  attempt (and cap that attempt's timeout by the time left), before and after
  every search, and before the final answer is accepted. When a limit is hit,
  the run stops instead of forcing out an answer we could not verify, and the
  trace records which limit it was.
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
# The UI only knows three answer modes plus ERROR, so several stop reasons
# share the same mode. We still keep the detailed stop reason in the trace
# so the evaluation can tell them apart.
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
}


# ================================
# Per-request state
# ================================
class AgentRun:
    """
    All the state for ONE question, shared by every agent that works on it.

    Why we need this:
    - Workers run in their own threads, so we update the counters and the
      trace under a lock.
    - Nothing in here is shared between different user requests. Each call
      to AgentAssistant.query() starts with a fresh AgentRun.
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
        # IMPORTANT: Only the thread that created the run may call on_event.
        # The Streamlit callback fails when it is called from a worker
        # thread, so worker events wait in a queue until the main thread
        # flushes them.
        self._owner_thread = threading.get_ident()
        self._pending_events = []

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
        """
        Append a step to the trace and forward it to the UI callback.

        Events recorded on a worker thread are only queued here. The main
        thread sends them later through flush_events().
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
    """
    The OLED agent: plan, route, research with tools, review, and answer.

    It searches through one shared Retriever, so the embedding model, the
    vector store, and the reranker are only loaded once.
    """

    def __init__(self, retriever):
        self.retriever = retriever
        self.search_tool = SearchTool(retriever)
        # max_retries=0 turns off the OpenAI client's own retries. respond()
        # retries transient errors itself, because that way we can cap every
        # attempt's timeout by the time left in the request. The client's
        # built-in retries don't know about our deadline and could quietly
        # use it all up.
        self.client = OpenAI(timeout=config.AGENT_API_TIMEOUT_SECONDS, max_retries=0)
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
        """
        Make one Responses API call on behalf of `role`.

        Why we need this:
        Every LLM call in the agent goes through this method, so this is the
        one place where we count the call against the budget, check the
        deadline, retry transient errors, and record token usage.

        Args:
            run: The AgentRun for the current question.
            role: Who is calling (e.g., "planner", "worker1", "reviewer").
                Used for the per-role call counts and the trace.
            model: OpenAI model name to call.
            input_items: Conversation items sent as the request `input`.
            tools: Optional tool schemas. When given, the model must call a tool.
            text_format: Optional structured output format (JSON schema).

        Returns:
            The Responses API response object.

        Note:
            store=False keeps our requests out of OpenAI's response storage.
            Instead, we pass the encrypted reasoning items back to the model
            in the next turn. Budget, deadline, and API problems are raised
            as AgentStop.
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

        # We retry transient errors, but every attempt (and the pause before
        # it) has to fit in the time left in the request.
        retry_pause = 1.0
        for attempt in range(config.AGENT_API_RETRIES + 1):
            run.check_deadline(f"{role} call")
            kwargs["timeout"] = min(config.AGENT_API_TIMEOUT_SECONDS, run.time_left())
            try:
                response = self.client.responses.create(**kwargs)
                break
            except APITimeoutError as exc:
                # If the timeout only happened because we capped it to the
                # deadline, we report it as a deadline stop.
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
            # A broken plan shouldn't block the question, so we fall back to
            # an "uncertain" plan and search the question as it was asked.
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
        """
        Run one search_documents call for `role` and return the tool output.

        seen_chunk_ids holds the chunks this context has already received in
        full (see search_result_for_model), so we don't send their text twice.
        """
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return {"status": "error", "message": "query must be a non-empty string."}
        query = query.strip()
        if len(query) > config.AGENT_MAX_QUERY_CHARS:
            return {"status": "error", "message": f"query is longer than {config.AGENT_MAX_QUERY_CHARS} characters."}

        # A repeated query would return the same results, so we answer it from
        # the cache without spending a search. If another agent ran the query,
        # this context still gets the chunk text it hasn't seen yet.
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
        # Searches can wait for each other on the shared CPU lock, so we check
        # the deadline again once the search is done.
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
        """
        Tool-calling loop shared by the single agent, the workers, the
        orchestrator, and an escalated heavy run.

        Each turn we call the model with the tools it may use right now, run
        the tool calls it makes, and send the results back. The loop ends
        when a finish tool handler says we are done.

        Args:
            run: The AgentRun for the current question.
            role: Who is running the loop (used for budgets and the trace).
            model: OpenAI model name for this loop.
            instructions: Developer prompt for this loop.
            user_content: The first user message (question, plan, evidence).
            finish_tools: The tools that can end this loop (submit / declare).
            handle_finish: Called as handle_finish(name, args). It returns
                ("done", value) to end the loop with value, or
                ("continue", tool_output) to send feedback to the model.
                It may also raise AgentStop to end the whole run.
            given_chunk_ids: Chunks whose text is already in user_content.
                If there are none, the loop has to search before it is
                allowed to finish.

        Returns:
            The value that handle_finish returned together with "done".
        """
        input_items = [
            {"role": "developer", "content": instructions},
            {"role": "user", "content": user_content},
        ]
        finish_names = {tool["name"] for tool in finish_tools}
        # Chunks whose full text this context has received. These are also
        # the ids it is allowed to cite.
        seen_chunk_ids = set(given_chunk_ids)
        require_search = not seen_chunk_ids
        calls_used = 0
        searched_here = False

        while True:
            if calls_used >= config.AGENT_MAX_CALLS_PER_LOOP:
                raise AgentStop("llm_budget_exhausted", f"{role} used its {calls_used} calls")

            # On the last call this loop may make, we take the search tool
            # away so the model has to finish with what it has. Its answer is
            # still validated as usual.
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
            # We keep the model's own output (including the encrypted
            # reasoning) in the context, so the next turn picks up the same
            # line of thought.
            input_items.extend(item.model_dump(exclude_none=True) for item in response.output)
            calls = [item for item in response.output if item.type == "function_call"]

            # The model replied in plain text without calling a tool, so we
            # count it as an invalid turn and remind it to use a tool.
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
                        # If the model finished in the same turn as it
                        # searched, it would never read the new results.
                        result = {"status": "error", "message": f"{call.name} must be called alone, after reading the search results."}
                    else:
                        outcome, payload = handle_finish(call.name, args)
                        if outcome == "done":
                            return payload
                        result = payload
                        if result.get("status") == "rejected":
                            # Models sometimes miscopy an 8-character hex id.
                            # Showing the valid ids lets them fix it in one turn.
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
        """
        Build the submit_answer / declare_insufficient handler for `role`.

        For submit_answer we run the citation checks first. If they pass and
        the reviewer is on, the answer is reviewed claim by claim before we
        accept it. declare_insufficient ends the run with insufficient_evidence.
        """
        state = {"revisions": 0}

        def handle(name, args):
            if name == "declare_insufficient":
                missing = str(args.get("missing", "")).strip()[: config.AGENT_MAX_QUERY_CHARS]
                raise AgentStop("insufficient_evidence", missing)

            answer = args.get("answer")
            citations = args.get("citations")
            # 1) Citation checks. A rejected answer goes back to the writer
            #    until it runs out of revisions.
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
            # 3) The deadline covers the whole request, including the moment
            #    we accept the final answer.
            run.check_deadline(f"{role} final answer")
            return "done", (answer, citations)

        return handle

    # ---------- single agent (light, heavy, or the only model) ----------
    def answer_alone(self, run, role, model, prior_evidence=None):
        """
        Let one agent research and answer the whole question.

        prior_evidence holds the chunks an earlier agent already collected in
        this request (we use it on escalation). We pass them in as original
        text, so the loop is allowed to finish without searching again.
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
        """
        Decide whether a failed light run should be retried on the heavy model.

        We escalate at most once, and only for failures that a stronger model
        could plausibly fix: citations the light model could not get right, a
        loop that ran out of turns, or "insufficient evidence" even though
        retrieval scored high (the evidence was probably there, but the light
        model misread it). Out-of-domain questions, API errors, and the global
        deadline never escalate.
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
            # The heavy model starts in a fresh context. It gets the question,
            # the plan, and the original chunk text, but none of the light
            # model's conversation.
            return self.answer_alone(run, "heavy", self.heavy_model, prior_evidence=run.ledger.entries())

    # ---------- Public entry point ----------
    def query(self, question, on_event=None):
        """
        Answer one question with the agent.

        Args:
            question: The user's question.
            on_event: Optional callback that receives each trace event as it
                happens (e.g., the Streamlit UI). It is only ever called from
                the main thread.

        Returns:
            dict: "answer", "mode", "relevance_score", "retrieval_metadata",
            "sources", and "agent" (plan, trace, usage, evidence).
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
                "question": question,
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
