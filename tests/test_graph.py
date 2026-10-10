"""Real LangGraph/SQLite, scripted LLM responses; no paid calls or model downloads."""
import json
from types import SimpleNamespace

import pytest

import config
from agent_graph import GraphAgentAssistant
from agent_runtime import AgentAssistant
from agent_tools import make_chunk_id


DOC = SimpleNamespace(page_content="FIrpic is a blue phosphorescent emitter.", metadata={"source": "fixture.pdf", "page": 0})
CID = make_chunk_id(DOC)


class Retriever:
    min_doc_relevance = .5
    def __init__(self): self.queries = []
    def retrieve_candidates(self, query):
        self.queries.append(query)
        return [{"doc": DOC, "relevance": .9}]
    def filter_candidates_by_relevance(self, candidates): return candidates
    def rerank_candidates(self, query, candidates): return candidates, False


class Call(SimpleNamespace):
    def model_dump(self, **kwargs): return vars(self)


def tool(name, **args):
    return SimpleNamespace(output_text="", usage=None, output=[Call(
        type="function_call", name=name, arguments=json.dumps(args), call_id="call_fixture")])


def plan(domain="in_domain", complexity="simple", subquestions=None, **extra):
    return SimpleNamespace(usage=None, output=[], output_text=json.dumps({
        "domain": domain, "reason": "fixture", "complexity": complexity,
        "subquestions": subquestions or ["FIrpic emitter"], "sequential": False, **extra}))


def review(supported=True, core=True):
    return SimpleNamespace(usage=None, output=[], output_text=json.dumps({
        "core_question_answered": core,
        "claims": [{"claim": "FIrpic is a blue emitter", "supported": supported,
                    "fix": "none" if supported else "research", "reason": "fixture"}]}))


def answer(cid=CID):
    return tool("submit_answer", answer=f"FIrpic is a blue phosphorescent emitter. [{cid}]", citations=[cid])


class Client:
    def __init__(self, replies):
        self.replies = list(replies); self.requests = []
        self.responses = self
    def create(self, **kwargs):
        # Snapshot inputs before the runtime appends subsequent turns.
        self.requests.append(json.loads(json.dumps(kwargs)))
        if not self.replies: raise AssertionError("Unexpected LLM call")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception): raise reply
        return reply


@pytest.fixture(autouse=True)
def defaults(monkeypatch):
    monkeypatch.setattr(config, "AGENT_ROUTING", True)
    monkeypatch.setattr(config, "AGENT_REVIEWER", True)
    monkeypatch.setattr(config, "AGENT_WORKERS", 2)
    monkeypatch.setattr(config, "SESSION_MEMORY_ENABLED", True)


def normalize_requests(requests):
    return [{k: v for k, v in r.items() if k != "timeout"} for r in requests]


@pytest.mark.parametrize("replies,mode", [
    ([plan(), tool("search_documents", query="FIrpic"), answer(), review()], "RAG"),
    ([plan(complexity="complex"), tool("search_documents", query="FIrpic"), answer(), review()], "RAG"),
    ([plan(domain="out_of_domain")], "OFF_TOPIC"),
    ([plan(), tool("search_documents", query="FIrpic"), answer("c00000000"), answer(), review()], "RAG"),
    ([plan(), tool("search_documents", query="FIrpic"), answer(), review(False), answer(), review()], "RAG"),
    ([plan(), tool("search_documents", query="FIrpic"), tool("declare_insufficient", missing="details"), answer(), review()], "RAG"),
])
def test_graph_matches_legacy_call_sequence(replies, mode):
    old_client, new_client = Client(replies), Client(replies)
    old = AgentAssistant(Retriever(), client=old_client).query("What is FIrpic?")
    new = GraphAgentAssistant(Retriever(), client=new_client).query("What is FIrpic?")
    assert old["mode"] == new["mode"] == mode
    for key in ("answer", "sources", "retrieval_metadata"):
        assert old[key] == new[key]
    assert normalize_requests(old_client.requests) == normalize_requests(new_client.requests)


def test_sqlite_history_isolation_and_reopen(tmp_path):
    path = str(tmp_path / "memory.sqlite")
    client = Client([plan(), tool("search_documents", query="FIrpic"), answer(), review()])
    a = GraphAgentAssistant.with_sqlite(Retriever(), path, client)
    first = a.query("What is FIrpic?", thread_id="a")
    a.close()
    follow = plan(standalone_question="Why is FIrpic unstable?", needs_clarification=False, clarification_question="")
    client = Client([follow, tool("search_documents", query="FIrpic instability"), answer(), review(), plan(domain="out_of_domain")])
    b = GraphAgentAssistant.with_sqlite(Retriever(), path, client)
    second = b.query("Why is it unstable?", thread_id="a")
    assert second["standalone_question"] == "Why is FIrpic unstable?"
    assert first["agent"]["searches"] == second["agent"]["searches"] == 1
    assert first["agent"]["usage"]["llm_calls"] == second["agent"]["usage"]["llm_calls"] == 4
    planner_input = json.loads(client.requests[0]["input"][1]["content"])
    assert len(planner_input["recent_turns"]) == 1
    # Writer receives only the resolved question/plan and freshly searched evidence.
    assert "Why is it unstable?" not in json.dumps(client.requests[1]["input"])
    b.query("Bake a cake", thread_id="b")
    assert client.requests[-1]["text"]["format"]["name"] == "plan"
    assert len(b.session("a")["history"]) == 2
    assert len(b.session("b")["history"]) == 1
    b.close()


def test_clarification_keeps_original_intent(tmp_path):
    replies = [plan(), tool("search_documents", query="FIrpic"), answer(), review(),
        plan(standalone_question="Compare FIrpic with another emitter", needs_clarification=True,
             clarification_question="Which emitter should I compare with FIrpic?"),
        plan(standalone_question="Compare FIrpic and Ir(ppy)3", needs_clarification=False, clarification_question=""),
        tool("search_documents", query="FIrpic Ir(ppy)3"), answer(), review()]
    client = Client(replies)
    a = GraphAgentAssistant.with_sqlite(Retriever(), str(tmp_path / "s.sqlite"), client)
    a.query("What is FIrpic?", thread_id="a")
    c = a.query("Compare it with another emitter", thread_id="a")
    assert c["mode"] == "CLARIFICATION" and c["agent"]["searches"] == 0
    assert a.session("a")["pending_clarification"]["question"] == "Compare FIrpic with another emitter"
    result = a.query("Ir(ppy)3", thread_id="a")
    assert result["standalone_question"] == "Compare FIrpic and Ir(ppy)3"
    assert a.session("a")["pending_clarification"] is None
    a.close()


def test_stream_events_and_five_turn_window(tmp_path):
    replies = [plan(domain="out_of_domain")]
    replies += [plan(domain="out_of_domain", standalone_question=f"Bake cake {i}", needs_clarification=False,
                     clarification_question="") for i in range(1, 7)]
    a = GraphAgentAssistant.with_sqlite(Retriever(), str(tmp_path / "s.sqlite"), Client(replies))
    for i in range(7):
        events = list(a.stream(f"Bake cake {i}", thread_id="a"))
        assert events[-1]["type"] == "result"
        assert any(x["type"] == "event" and x["data"]["type"] == "plan" for x in events)
    assert len(a.session("a")["history"]) == 5
    assert a.session("a")["history"][0]["user"] == "Bake cake 2"
    a.close()


def test_malformed_rewrite_fails_closed_and_history_survives(tmp_path):
    client = Client([plan(domain="out_of_domain"), SimpleNamespace(output_text="{}", usage=None)])
    a = GraphAgentAssistant.with_sqlite(Retriever(), str(tmp_path / "s.sqlite"), client)
    a.query("Bake cake", thread_id="a")
    r = a.query("What about that?", thread_id="a")
    assert r["mode"] == "ERROR" and r["agent"]["searches"] == 0
    assert len(a.session("a")["history"]) == 1
    a.close()


def test_history_cannot_supply_citations(tmp_path):
    client = Client([plan(), tool("search_documents", query="FIrpic"), answer(), review()])
    a = GraphAgentAssistant.with_sqlite(Retriever(), str(tmp_path / "s.sqlite"), client)
    a.query("What is FIrpic?", thread_id="a")
    # The first tool offered on a new follow-up is search only, despite history.
    client.replies = [plan(standalone_question="FIrpic stability", needs_clarification=False, clarification_question=""),
                      tool("submit_answer", answer=f"Old fact [{CID}]", citations=[CID])]
    r = a.query("Its stability?", thread_id="a")
    assert [t["name"] for t in client.requests[5]["tools"]] == ["search_documents"]
    assert r["mode"] != "RAG" and not r["sources"]
    a.close()


def test_refusal_keeps_retrieved_evidence_for_private_diagnostics(monkeypatch):
    """A refusal has no cited evidence but retains the chunks the agent inspected."""
    monkeypatch.setattr(config, "AGENT_ROUTING", False)
    client = Client([
        plan(),
        tool("search_documents", query="FIrpic"),
        tool("declare_insufficient", missing="required lifetime data"),
    ])

    result = AgentAssistant(Retriever(), client).query("What is FIrpic's lifetime?")

    assert result["mode"] == "NO_ANSWER_IN_DOCS"
    assert result["agent"]["evidence"] == []
    assert result["agent"]["retrieved_evidence"][0]["chunk_id"] == CID


def test_team_review_feedback_never_restarts_workers():
    replies = [plan(complexity="complex", subquestions=["emitter identity", "emitter properties"], sequential=True),
               tool("search_documents", query="identity"), tool("submit_findings", findings=f"FIrpic [{CID}]", citations=[CID]),
               tool("search_documents", query="properties"), tool("submit_findings", findings=f"Blue emitter [{CID}]", citations=[CID]),
               answer(), review(False), answer(), review()]
    old_client, new_client = Client(replies), Client(replies)
    old = AgentAssistant(Retriever(), old_client).query("Two questions")
    new = GraphAgentAssistant(Retriever(), new_client).query("Two questions")
    assert old["mode"] == new["mode"] == "RAG"
    assert normalize_requests(old_client.requests) == normalize_requests(new_client.requests)
    assert sum(e["type"] == "worker_start" for e in new["agent"]["trace"]) == 2
    assert new["agent"]["review_rounds"] == 2


@pytest.mark.parametrize("setting,value,replies,reason", [
    ("AGENT_MAX_LLM_CALLS", 2, [plan(), tool("search_documents", query="FIrpic")], "llm_budget_exhausted"),
    ("AGENT_DEADLINE_SECONDS", 0, [], "deadline_exceeded"),
])
def test_request_limits_survive_graph_boundaries(monkeypatch, setting, value, replies, reason):
    monkeypatch.setattr(config, setting, value)
    for cls in [AgentAssistant, GraphAgentAssistant]:
        result = cls(Retriever(), Client(replies)).query("FIrpic?")
        assert result["agent"]["stop_reason"] == reason
        assert result["mode"] == "NO_ANSWER_IN_DOCS"


def test_stateless_calls_do_not_rewrite_even_with_checkpointer(tmp_path):
    client = Client([plan(domain="out_of_domain"), plan(domain="out_of_domain")])
    a = GraphAgentAssistant.with_sqlite(Retriever(), str(tmp_path / "s.sqlite"), client)
    a.query("Cake?")
    a.query("Another recipe?")
    assert all(r["text"]["format"]["name"] == "plan" for r in client.requests)
    a.close()
