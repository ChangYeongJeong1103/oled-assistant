"""
AI-Driven OLED Assistant - Streamlit Application

Every question goes through GraphAgentAssistant with per-session memory.
"""
import streamlit as st
import time
import os
from retrieval import build_retriever
from agent_runtime import save_trace
from agent_graph import GraphAgentAssistant
from uuid import uuid4
import config
from utils import format_time

# Page Configuration
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide"
)


def format_source_line(source):
    """Format one source as a Markdown line such as `"[1] Paper title — p.3"`.

    The title becomes a link when the registry has a verified URL for it.
    Titles and URLs always come from the source registry, so the model cannot create them.
    """
    # Square brackets inside a title would break the markdown link syntax.
    title = source["title"].replace("[", "(").replace("]", ")")
    label = f"[{title}]({source['url']})" if source.get("url") else title
    page = f" — p.{source['page']}" if source.get("page") else ""
    return f"**[{source['number']}]** {label}{page}"


def format_answer_markdown(answer):
    """Prefix the answer with a bold `"Answer:"` label.

    Markdown tables, lists and headings must begin on their own line, so block content appears below the label.
    """
    starts_with_block = answer.lstrip().startswith(("|", "- ", "* ", "#"))
    separator = "\n\n" if starts_with_block else " "
    return f"**Answer:**{separator}{answer}"


def render_trace(trace):
    """Render a short and readable version of the agent trace.

    The trace shows the plan, route, searches, worker and review steps and stop reason.
    Each line begins with the role that produced it, such as planner, worker1, orchestrator or reviewer.
    """
    for event in trace:
        event_type = event["type"]
        # The role label is empty for run-level events such as `stop`.
        who = f"`{event['role']}` " if event.get("role") else ""

        if event_type == "plan":
            st.markdown(
                f"**Plan** · {who}`{event.get('model', '')}` · domain=`{event['domain']}`, "
                f"complexity=`{event.get('complexity')}`"
                + (" (sequential)" if event.get("sequential") else "")
                + f" — {event['reason']}"
            )
            for index, subquestion in enumerate(event["subquestions"], 1):
                st.markdown(f"&nbsp;&nbsp;{index}. {subquestion}")
        elif event_type == "interpretation":
            st.markdown(f"**Interpreted as:** {event['standalone_question']}")
        elif event_type == "route":
            st.markdown(f"🧭 Route: `{event['route']}`")
        elif event_type == "escalate":
            st.markdown(
                f"⬆️ Escalated `{event['from_model']}` → `{event['to_model']}` "
                f"after `{event['reason']}` (fresh context, evidence kept)"
            )
        elif event_type == "parallel_calls":
            st.markdown(f"⏩ {who}{len(event['tools'])} tool calls in one turn")
        elif event_type == "search":
            st.markdown(
                f"🔎 {who}`{event['query']}` → {event['result_count']} chunks "
                f"({event['new_chunks']} new), max relevance {event['max_relevance']:.3f}"
            )
        elif event_type == "search_repeat":
            st.markdown(f"↩️ {who}Repeated query skipped: `{event['query']}`")
        elif event_type == "worker_start":
            st.markdown(f"👷 {who}`{event['model']}` started: {' / '.join(event['subquestions'])}")
        elif event_type == "worker_done":
            st.markdown(f"👷 {who}finished: `{event['status']}`, {event['citations']} cited chunks")
        elif event_type == "orchestrate":
            st.markdown(f"🧩 {who}`{event['model']}` checks {event['chunks']} cited chunks and writes the answer")
        elif event_type == "review":
            st.markdown(
                f"🧐 {who}round {event['round']}: {event['unsupported']}/{event['claims']} claims "
                f"unsupported, core question answered: {event['core_question_answered']}"
            )
            for claim in event.get("unsupported_claims", []):
                st.markdown(f"&nbsp;&nbsp;– {claim}")
        elif event_type in ("answer_rejected", "findings_rejected"):
            st.markdown(f"✏️ {who}sent back for revision: {'; '.join(event['errors'])}")
        elif event_type == "api_retry":
            st.markdown(f"🔁 {who}API call retried after `{event['reason']}`")
        elif event_type == "stop":
            st.markdown(f"**Stop reason:** `{event['reason']}` at {event['t']}s")
        else:
            st.markdown(f"⚠️ {who}{event_type}")


def render_message_extras(metadata=None, sources=None, trace=None):
    """Render sources and the `"Agent Trace"` and `"Analysis Details"` expanders.

    Streamlit draws chat messages through the history loop on every rerun and through the input block immediately after submission.
    A new message would appear without expanders until the next rerun if only the history path rendered them.
    This shared function keeps both paths identical and prevents that first-render issue.
    """
    if sources:
        # Answers cite `[n]` inline, so sources appear directly below the answer.
        st.markdown("**Sources**")
        for source in sources:
            st.markdown(format_source_line(source))
    if trace:
        with st.expander("Agent Trace"):
            render_trace(trace)
    if metadata:
        with st.expander("Analysis Details"):
            st.json(metadata)

# Cache the agent so the embedding model, vector store and reranker load once per server process.
@st.cache_resource
def get_agent():
    return GraphAgentAssistant.with_sqlite(build_retriever())


try:
    if not os.environ.get("OPENAI_API_KEY"):
        st.error(
            "❌ OPENAI_API_KEY not found. Set it in your environment or `.env` file before running the app."
        )
        st.stop()
    assistant = get_agent()
except Exception as e:
    st.error(f"Failed to initialize the agent: {str(e)}")
    st.stop()

# Sidebar
with st.sidebar:
    st.title(f"{config.APP_ICON} OLED Assistant")
    st.markdown("---")
    st.markdown("**System Status**")
    features = assistant.features()
    if features["routing"]:
        st.success(f"Models: {features['light_model']} (light) / {features['heavy_model']} (heavy)")
    else:
        st.success(f"Model: {features['light_model']}")
    # Show which optional agent features are switched on (set by env flags).
    st.info(
        "Features: "
        f"parallel search {'on' if features['parallel_search'] else 'off'} · "
        f"routing {'on' if features['routing'] else 'off'} · "
        f"workers {features['workers'] or 'off'} · "
        f"reviewer {'on' if features['reviewer'] else 'off'}"
    )
    st.info(f"Retrieval: candidates={config.CANDIDATE_TOP_K}, final={config.FINAL_TOP_N}")
    st.info(f"Doc relevance filter ≥ {config.MIN_DOC_RELEVANCE}")
    st.info("Session memory: last 5 turns" if config.SESSION_MEMORY_ENABLED else "Session memory: off")
    st.info(
        f"Budget: {config.AGENT_MAX_SEARCHES} searches, "
        f"{config.AGENT_MAX_LLM_CALLS} LLM calls, {config.AGENT_DEADLINE_SECONDS}s"
    )
    st.markdown("---")
    st.markdown("### User Guide")
    st.markdown("""
    1. Ask questions about **OLED physics, materials, or fabrication**.
    2. The agent first **plans**: is the question about OLEDs, and does it
       need to be split into sub-questions?
    3. It then **searches** (up to the budget), rewording the query when
       results are weak, and answers only with **cited** evidence.
    4. If the documents don't cover it, it says so instead of guessing.
    """)

# Initialize chat history (must come before any UI that reads it).
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------------------------------------------------------
# Resolve the prompt first, add its user message to history and then render the page.
# This order makes the `"New chat"` button visible as soon as a user sends the first message.
# It also hides example cards on the same render that starts processing the new query.
# ----------------------------------------------------------------------------
typed_prompt = st.chat_input("Ask a question about OLED technology...")
prompt = typed_prompt or st.session_state.pop("pending_query", None)

# Add the user message before rendering anything that depends on conversation state.
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

# Main Chat Interface
# Display the title on the left and the `"New chat"` button on the right.
title_col, action_col = st.columns([5, 1])
with title_col:
    st.title(config.APP_TITLE)
    st.markdown("#### Intelligent Engineering Support System")
with action_col:
    # Vertical spacer so the button sits roughly aligned with the subtitle.
    st.write("")
    st.write("")
    # Show only when a conversation exists (welcome screen stays clean).
    if st.session_state.messages:
        if st.button("🔄 New chat", use_container_width=True, key="reset_chat_btn"):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid4())
            st.session_state.pop("pending_query", None)
            st.rerun()


# ----------------------------------------------------------------------------
# The welcome screen groups example queries by answer mode.
# The agent answers only from cited documents and refuses everything else.
# Examples help new users understand document coverage and each RAG, NO_ANSWER and OFF_TOPIC outcome.
# Each example button stores `st.session_state.pending_query` and reruns the page.
# The chat-input block handles that value like a typed prompt to avoid duplicating the answer pipeline.
# ----------------------------------------------------------------------------
EXAMPLE_QUERIES = {
    "🟢 RAG Mode": {
        "caption": "Document-grounded technical answer",
        "queries": [
            "What is OLED operation principle?",
            "What is the role of the hole transport layer in OLED devices?",            
            "What are typical electron mobility values in OLED ETL materials?",
        ],
    },
    "🟠 NO_ANSWER Mode": {
        "caption": "On-topic, but the docs don't cover it",
        "queries": [
            "What is the supply chain cost of phosphorescent OLEDs?",
            "Which company filed the most OLED patents last year?",
        ],
    },
    "🔴 OFF_TOPIC Mode": {
        "caption": "Rejected at the planning step, before any search",
        "queries": [
            "How do I bake a chocolate cake?",
            "Recommend me a Netflix show.",
        ],
    },
}

# ----------------------------------------------------------------------------
# Welcome screen container.
#
# Wrapping the welcome screen in `st.empty()` ensures Streamlit clears its DOM content when `show_welcome` is false.
# Without the wrapper, example cards from the previous render may remain visible while a long answer is generated.
# ----------------------------------------------------------------------------
welcome_slot = st.empty()

# Welcome cards appear only when no conversation exists.
# Because an incoming message is already in history, it hides the welcome screen immediately.
show_welcome = not st.session_state.messages

if show_welcome:
    with welcome_slot.container():
        st.markdown("### Try these example queries")
        st.caption(
            "Try a question, then ask a follow-up in the same chat. "
            "Open the Agent Trace under each answer to see its plan and searches."
        )
        # 3 columns so users can compare the three tiers at a glance.
        columns = st.columns(len(EXAMPLE_QUERIES))
        for col, (mode_label, payload) in zip(columns, EXAMPLE_QUERIES.items()):
            with col:
                st.markdown(f"**{mode_label}**")
                st.caption(payload["caption"])
                for j, query_text in enumerate(payload["queries"]):
                    button_key = f"example_{mode_label}_{j}"
                    if st.button(query_text, key=button_key, use_container_width=True):
                        # Store the query and rerun so the prompt resolver uses it on the next render.
                        st.session_state.pending_query = query_text
                        st.rerun()
        st.markdown("---")

# Display Chat History
# This also renders the newly added user message, so a separate user-message block is unnecessary.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_message_extras(
            metadata=message.get("metadata"),
            sources=message.get("sources"),
            trace=message.get("trace"),
        )


def run_agent_with_progress(question):
    """Run the agent and show each plan and search step in a live status box."""
    with st.status("Planning...", expanded=False) as status_box:

        def on_event(event):
            # The agent calls this after every step so users can follow multi-search progress.
            # Parallel-worker events arrive together after completion because only the main thread can update Streamlit.
            event_type = event["type"]
            role = event.get("role") or ""
            if event_type == "plan":
                status_box.update(label="Searching documents...")
                status_box.write(f"Plan ({event.get('complexity')}): {' / '.join(event['subquestions'])}")
            elif event_type == "route":
                status_box.write(f"Route: {event['route']}")
            elif event_type == "search":
                status_box.update(label=f"Searched {event['query']!r} — reading results...")
                status_box.write(f"🔎 [{role}] {event['query']} → {event['result_count']} chunks")
            elif event_type == "escalate":
                status_box.update(label=f"Escalating to {event['to_model']}...")
            elif event_type == "worker_start":
                status_box.update(label="Workers researching sub-questions...")
            elif event_type == "orchestrate":
                status_box.update(label="Orchestrator verifying findings...")
            elif event_type == "review":
                status_box.update(label="Reviewer checking claims...")
                status_box.write(f"🧐 Review {event['round']}: {event['unsupported']}/{event['claims']} unsupported")
            elif event_type in ("answer_rejected", "findings_rejected"):
                status_box.update(label="Fixing citations...")
            elif event_type == "interpretation":
                status_box.write(f"Interpreted as: {event['standalone_question']}")

        result = None
        for item in assistant.stream(question, thread_id=st.session_state.thread_id):
            if item["type"] == "event":
                on_event(item["data"])
            else:
                result = item["data"]
        if result is None:
            raise RuntimeError("No graph result received")
        status_box.update(label=f"Done ({result['agent']['stop_reason']})",
                          state="error" if result["mode"] == "ERROR" else "complete")
    save_trace(result)
    return result


if prompt:
    # Generate the assistant response (user message was already added above).
    with st.chat_message("assistant"):
        start_time = time.time()

        result = run_agent_with_progress(prompt)
        # Create the answer slot after the status box so it renders below it.
        message_placeholder = st.empty()

        elapsed = time.time() - start_time
        
        # Format output based on mode
        answer = result["answer"]
        mode = result["mode"]
        score = result["relevance_score"]
        
        # Color-coded status
        if mode == "RAG":
            status_color = "green"
            icon = "🟢"
            mode_text = "RAG Mode (Document-based Response)"
        elif mode == "NO_ANSWER_IN_DOCS":
            status_color = "orange"
            icon = "🟠"
            mode_text = "No Answer (No Relevant Content in Documents)"
        elif mode == "OFF_TOPIC":
            status_color = "red"
            icon = "🔴"
            mode_text = "Off-Topic Rejection (Classified out of domain)"
        elif mode == "ERROR":
            status_color = "gray"
            icon = "⚠️"
            mode_text = "Generation Error"
        elif mode == "CLARIFICATION":
            status_color = "blue"
            icon = "💬"
            mode_text = "Clarification needed"
        else:
            status_color = "gray"
            icon = "⚪"
            mode_text = f"Unknown mode: {mode}"

        # `score` is the relevance of the single strongest retrieved document.
        status_text = f"{icon} **{mode_text}** | Top Relevance: {score:.3f}"
        searches = result["agent"]["searches"]
        # An out-of-domain question stops before search, so show the search count instead of a misleading zero relevance.
        if searches == 0:
            status_text = f"{icon} **{mode_text}** | Searches: 0"
        else:
            status_text += f" | Searches: {searches}"

        # Display Answer (the same text is saved to the chat history below)
        answer_content = (
            f"{format_answer_markdown(answer)}\n\n"
            f":{status_color}[{status_text}] | Time: {format_time(elapsed)}"
        )
        if result["standalone_question"] != result["original_question"]:
            answer_content = f"**Interpreted as:** {result['standalone_question']}\n\n" + answer_content
        message_placeholder.markdown(answer_content)

        # Build the metadata and source payload once so live output and saved history remain identical.
        live_metadata = {
            "mode": mode,
            "response_time": f"{elapsed:.2f}s",
            "retrieval": result.get("retrieval_metadata", {}),
        }
        # Store sources as plain dictionaries instead of Document objects to keep chat history small.
        live_sources = result["sources"] if mode == "RAG" else None
        live_trace = result["agent"]["trace"]

        # Render expanders immediately instead of waiting for the next Streamlit rerun.
        render_message_extras(
            metadata=live_metadata,
            sources=live_sources,
            trace=live_trace,
        )

        # Save to history so the same message keeps showing on later reruns.
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_content,
            "metadata": live_metadata,
            "sources": live_sources,
            "trace": live_trace,
        })
