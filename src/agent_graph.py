"""LangGraph controls routing and escalation while each request keeps independent Strict-RAG state.

Review and revision remain in the existing research loop, so reviewer feedback does not restart workers.
"""
import sqlite3
import threading
from pathlib import Path
from typing import TypedDict
from weakref import WeakValueDictionary

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

import config
from agent_runtime import AgentAssistant, AgentRun
from agent_team import answer_with_team
from agent_tools import AgentStop
from session_memory import bounded_history, plan_with_history
from utils import logger


class GraphState(TypedDict, total=False):
    original_question: str
    history: list[dict]
    pending_clarification: dict | None
    run_data: dict
    answer: str | None
    citations: list[str]
    stop_reason: str
    stop_detail: str
    result: dict | None


class GraphAgentAssistant(AgentAssistant):
    def __init__(self, retriever, client=None, checkpointer=None):
        super().__init__(retriever, client=client)
        self._connection = None
        self.checkpointer = checkpointer
        self._locks = WeakValueDictionary()
        self._locks_guard = threading.Lock()
        self.graph = self._build_graph().compile()
        self.session_graph = self._build_graph().compile(checkpointer=checkpointer) if checkpointer else None

    @classmethod
    def with_sqlite(cls, retriever, path=config.SESSION_DB_PATH, client=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, check_same_thread=False, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        try:
            agent = cls(retriever, client=client, checkpointer=SqliteSaver(connection))
        except Exception:
            connection.close()
            raise
        agent._connection = connection
        return agent

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _thread_lock(self, thread_id):
        with self._locks_guard:
            lock = self._locks.get(thread_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[thread_id] = lock
            return lock

    def features(self):
        return {**super().features(), "orchestration": "langgraph",
                "session_memory": config.SESSION_MEMORY_ENABLED,
                "history_turns": config.SESSION_HISTORY_TURNS}

    def _build_graph(self):
        builder = StateGraph(GraphState)
        builder.add_node("initialize", self._initialize)
        builder.add_node("plan", self._node(self._plan_request))
        builder.add_node("route", self._node(self._route_request))
        builder.add_node("research", self._node(self._research_request))
        builder.add_node("team", self._node(self._team_request))
        builder.add_node("escalate", self._node(self._escalate_request))
        builder.add_node("finish", self._finish)
        builder.add_edge(START, "initialize")
        builder.add_edge("initialize", "plan")
        builder.add_conditional_edges("plan", self._after_plan, {"route": "route", "finish": "finish"})
        builder.add_conditional_edges("route", self._after_route,
                                      {"research": "research", "team": "team", "finish": "finish"})
        builder.add_conditional_edges("research", self._after_research,
                                      {"escalate": "escalate", "finish": "finish"})
        builder.add_edge("team", "finish")
        builder.add_conditional_edges("escalate", lambda s: "finish" if s["stop_reason"] else "research",
                                      {"research": "research", "finish": "finish"})
        builder.add_edge("finish", END)
        return builder

    def _initialize(self, state):
        # Reset request data while preserving completed conversation pairs and pending intent.
        run = AgentRun(state["original_question"], get_stream_writer())
        return {"run_data": run.snapshot(), "answer": None, "citations": [],
                "stop_reason": "", "stop_detail": "", "result": None,
                "history": bounded_history(state.get("history", [])),
                "pending_clarification": state.get("pending_clarification")}

    def _node(self, operation):
        def execute(state):
            run = AgentRun.restore(state["run_data"], get_stream_writer())
            try:
                updates = operation(run, state) or {}
            except AgentStop as stop:
                updates = {"stop_reason": stop.reason, "stop_detail": stop.detail}
            except Exception as exc:
                logger.exception("Graph node failed: %s", operation.__name__)
                updates = {"stop_reason": "api_error", "stop_detail": f"{type(exc).__name__}: {exc}"}
            run.flush_events()
            return {**updates, "run_data": run.snapshot()}
        return execute

    def _plan_request(self, run, state):
        history, pending = state.get("history", []), state.get("pending_clarification")
        if history or pending:
            plan = plan_with_history(self, run, history, pending)
        else:
            plan = self._plan(run)  # Original prompt/schema on no-history requests.
        if plan["domain"] == "out_of_domain":
            raise AgentStop("out_of_domain", plan.get("reason", ""))
        if plan.get("needs_clarification"):
            raise AgentStop("clarification_needed", plan["clarification_question"].strip())

    @staticmethod
    def _after_plan(state):
        return "finish" if state["stop_reason"] else "route"

    def _route_request(self, run, state):
        run.route = self._choose_route(run.plan)
        run.record("route", role="planner", route=run.route)

    @staticmethod
    def _after_route(state):
        if state["stop_reason"]:
            return "finish"
        return "team" if state["run_data"]["route"] == "team" else "research"

    def _research_request(self, run, state):
        role = "heavy" if run.escalated else {"single": "agent", "light": "light", "heavy": "heavy"}[run.route]
        model = self.heavy_model if role == "heavy" else self.light_model
        evidence = run.ledger.entries() if run.escalated else None
        answer, citations = self.answer_alone(run, role, model, prior_evidence=evidence)
        return {"answer": answer, "citations": citations, "stop_reason": "answered"}

    def _team_request(self, run, state):
        answer, citations = answer_with_team(self, run)
        return {"answer": answer, "citations": citations, "stop_reason": "answered"}

    def _after_research(self, state):
        if state["stop_reason"] == "answered":
            return "finish"
        run = AgentRun.restore(state["run_data"])
        stop = AgentStop(state["stop_reason"], state["stop_detail"])
        return "escalate" if self._should_escalate(run, stop) else "finish"

    def _escalate_request(self, run, state):
        run.check_deadline("escalation")
        run.escalated = True
        run.record("escalate", role="heavy", from_model=self.light_model, to_model=self.heavy_model,
                   reason=state["stop_reason"], detail=state["stop_detail"][:200])
        return {"stop_reason": "", "stop_detail": ""}

    def _finish(self, state):
        run = AgentRun.restore(state["run_data"], get_stream_writer())
        reason = state["stop_reason"] or "api_error"
        result = self.result_from_run(run, state["answer"], state["citations"], reason, state["stop_detail"])
        original = state["original_question"]
        result["original_question"] = original
        result["standalone_question"] = run.question
        result["agent"].update(original_question=original, standalone_question=run.question)
        pending = None
        if reason == "clarification_needed":
            result["answer"] = state["stop_detail"]
            pending = {"question": run.question, "asked": result["answer"]}
        # API failures do not erase pending clarification or useful history.
        history = state.get("history", [])
        if result["mode"] == "ERROR":
            pending = state.get("pending_clarification")
        else:
            history = bounded_history(history + [{"user": original, "standalone_question": run.question,
                                                   "assistant": result["answer"], "mode": result["mode"]}])
        return {"result": result, "history": history, "pending_clarification": pending,
                "run_data": {}, "answer": None, "citations": []}

    def stream(self, question, thread_id=None):
        """Yield progress events and the final result; omit `thread_id` for a stateless request."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        use_memory = thread_id is not None and config.SESSION_MEMORY_ENABLED
        if use_memory and self.session_graph is None:
            raise ValueError("A checkpointer is required for session memory")
        if use_memory and (not isinstance(thread_id, str) or not thread_id.strip()):
            raise ValueError("thread_id must be a non-empty string")
        graph = self.session_graph if use_memory else self.graph
        cfg = {"recursion_limit": 12}
        if use_memory:
            cfg["configurable"] = {"thread_id": thread_id}
        lock = self._thread_lock(thread_id) if use_memory else threading.Lock()
        with lock:
            for kind, data in graph.stream({"original_question": question}, cfg,
                                          stream_mode=["custom", "updates"]):
                if kind == "custom":
                    yield {"type": "event", "data": data}
                elif kind == "updates" and "finish" in data:
                    yield {"type": "result", "data": data["finish"]["result"]}

    def query(self, question, on_event=None, thread_id=None):
        result = None
        for item in self.stream(question, thread_id=thread_id):
            if item["type"] == "event" and on_event is not None:
                on_event(item["data"])
            elif item["type"] == "result":
                result = item["data"]
        if result is None:
            raise RuntimeError("Graph finished without a result")
        return result

    def session(self, thread_id):
        """Read bounded history to restore the UI for an existing trusted session."""
        if self.session_graph is None:
            return {"history": [], "pending_clarification": None}
        with self._thread_lock(thread_id):
            state = self.session_graph.get_state({"configurable": {"thread_id": thread_id}}).values
        return {"history": state.get("history", []), "pending_clarification": state.get("pending_clarification")}
