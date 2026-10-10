"""Streamlit UI regression using a real graph and scripted model responses."""
from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

import agent_graph
import agent_runtime
import retrieval
from test_graph import Client, Retriever, answer, plan, review, tool


def test_followup_display_and_new_chat(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    client = Client([
        plan(), tool("search_documents", query="FIrpic"), answer(), review(),
        plan(standalone_question="Compare FIrpic with another emitter",
             needs_clarification=True, clarification_question="Which emitter?"),
        plan(domain="out_of_domain"),
    ])
    agent = agent_graph.GraphAgentAssistant.with_sqlite(Retriever(), str(tmp_path / "ui.sqlite"), client)
    monkeypatch.setattr(retrieval, "build_retriever", lambda: Retriever())
    monkeypatch.setattr(agent_graph.GraphAgentAssistant, "with_sqlite", lambda *args, **kwargs: agent)
    monkeypatch.setattr(agent_runtime, "save_trace", lambda result: None)
    st.cache_resource.clear()
    app = AppTest.from_file(str(Path(__file__).parents[1] / "src" / "app.py"))
    try:
        app.run()
        assert not app.exception
        original_thread = app.session_state["thread_id"]
        app.chat_input[0].set_value("What is FIrpic?").run()
        assert not app.exception
        app.chat_input[0].set_value("Compare it with another emitter").run()
        assert not app.exception
        output = "\n".join(x.value for x in app.markdown)
        assert "Interpreted as:" in output
        assert "Clarification needed" in output
        assert "Which emitter?" in output
        app.button(key="reset_chat_btn").click().run()
        assert app.session_state["thread_id"] != original_thread
        assert app.session_state["messages"] == []
        app.chat_input[0].set_value("Bake cake").run()
        assert not app.exception
        assert client.requests[-1]["text"]["format"]["name"] == "plan"
    finally:
        st.cache_resource.clear()
        agent.close()
