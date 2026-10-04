"""
AI-Driven OLED Assistant - Streamlit Application
Aligned with OLED_assistant_v3_final.ipynb
"""
import streamlit as st
import time
import os
from rag_engine import StrictRAGAssistant, create_embeddings, get_vectorstore
from agent_runtime import AgentAssistant, save_trace
from source_registry import describe_source
import config
from utils import logger, format_time

IS_AGENT = config.ENGINE_MODE == "agent"

# Page Configuration
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide"
)


def sources_from_docs(docs):
    """
    Turn workflow Documents into the same source dicts the agent returns.

    Why we need this:
    The workflow result only carries LangChain Documents, while the agent
    already returns numbered source dicts. Converting them here lets the same
    rendering code show sources for both engines.
    """
    sources = []
    for number, doc in enumerate(docs, 1):
        sources.append({"number": number, **describe_source(doc.metadata)})
    return sources


def format_source_line(source):
    """
    Format one source as a markdown line, e.g. "[1] Paper title — p.3".

    The title becomes a link when the registry has a verified URL for it.
    NOTE: Titles and URLs always come from the source registry, so the model
    has no way to make them up.
    """
    # Square brackets inside a title would break the markdown link syntax.
    title = source["title"].replace("[", "(").replace("]", ")")
    label = f"[{title}]({source['url']})" if source.get("url") else title
    page = f" — p.{source['page']}" if source.get("page") else ""
    return f"**[{source['number']}]** {label}{page}"


def render_trace(trace):
    """
    Render a short, readable version of the agent trace.

    We show the plan, the routing decision, each search, the worker and review
    steps, and why the run stopped. Each line starts with the role that
    produced it (planner, worker1, orchestrator, reviewer, ...).
    """
    for event in trace:
        event_type = event["type"]
        # Role label, e.g. "`worker1` ". It stays empty for run-level events
        # such as stop.
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


def render_message_extras(metadata=None, sources=None, trace=None, sources_are_cited=False):
    """
    Render the source list and the "Agent Trace" / "Analysis Details" expanders.

    Why we need this helper:
    Streamlit redraws each chat message in TWO different code paths:
      1) the chat-history loop (runs on every page rerun)
      2) the chat-input block (runs ONCE, right after the user hits send)
    If the expanders only exist in path #1, the brand-new message a user just
    sent will look bare until the page reruns again. Centralising the render
    code here keeps both paths identical and avoids that "missing expander"
    bug on the very first render.
    """
    if sources:
        # Agent answers cite [n] inline, so we show the source list right
        # under the answer. Workflow answers have no inline citations, so we
        # keep their sources folded inside an expander.
        if sources_are_cited:
            st.markdown("**Sources**")
            for source in sources:
                st.markdown(format_source_line(source))
        else:
            with st.expander("Retrieved Documents"):
                for source in sources:
                    st.markdown(format_source_line(source))
    if trace:
        with st.expander("Agent Trace"):
            render_trace(trace)
    if metadata:
        with st.expander("Analysis Details"):
            st.json(metadata)

# Initialize Assistant (Cached to prevent reloading on every interaction)
@st.cache_resource
def get_assistant():
    embeddings = create_embeddings()
    vectorstore = get_vectorstore(embeddings)
    return StrictRAGAssistant(
        vectorstore=vectorstore,
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


@st.cache_resource
def get_agent():
    # The agent reuses the cached workflow for retrieval, so the embedding
    # model and reranker are loaded only once.
    return AgentAssistant(get_assistant())


try:
    if not os.environ.get("OPENAI_API_KEY"):
        st.error(
            "❌ OPENAI_API_KEY not found. Set it in your environment or `.env` file before running the app."
        )
        st.stop()
    # ENGINE_MODE=workflow keeps the original single-pass Strict RAG.
    assistant = get_agent() if IS_AGENT else get_assistant()
except Exception as e:
    st.error(f"Failed to initialize RAG Engine: {str(e)}")
    st.stop()

# Sidebar
with st.sidebar:
    st.title(f"{config.APP_ICON} OLED Assistant")
    st.markdown("---")
    st.markdown("**System Status**")
    if IS_AGENT:
        features = assistant.features()
        if features["routing"]:
            st.success(f"Models: {features['light_model']} (light) / {features['heavy_model']} (heavy)")
        else:
            st.success(f"Model: {features['light_model']}")
    else:
        st.success(f"Model: {config.LLM_MODEL}")
    st.info(f"Engine: {'Agent (plan → search loop)' if IS_AGENT else 'Workflow (single pass)'}")
    if IS_AGENT:
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
    if IS_AGENT:
        st.info(
            f"Budget: {config.AGENT_MAX_SEARCHES} searches, "
            f"{config.AGENT_MAX_LLM_CALLS} LLM calls, {config.AGENT_DEADLINE_SECONDS}s"
        )
    else:
        st.info(f"Off-topic gate < {config.OFF_TOPIC_THRESHOLD}")
    st.markdown("---")
    st.markdown("### User Guide")
    if IS_AGENT:
        st.markdown("""
        1. Ask questions about **OLED physics, materials, or fabrication**.
        2. The agent first **plans**: is the question about OLEDs, and does it
           need to be split into sub-questions?
        3. It then **searches** (up to the budget), rewording the query when
           results are weak, and answers only with **cited** evidence.
        4. If the documents don't cover it, it says so instead of guessing.
        """)
    else:
        st.markdown("""
        1. Ask questions about **OLED physics, materials, or fabrication**.
        2. Every retrieved document gets a **relevance** score (sigmoid-transformed).
        3. Documents above the filter are **reranked**, and the strongest ones become
           the **PRIMARY** source for the answer.
        4. If no document clears the filter, the system says so instead of guessing.
        """)

# Initialize chat history (must come before any UI that reads it).
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------------------------------------------------------
# Resolve the active prompt FIRST, then commit the user message to history,
# THEN render the page. This ordering matters for two reasons:
#   1) The "New chat" button in the title bar needs `messages` to already
#      include the new user message, otherwise it stays hidden until the
#      *next* page rerun.
#   2) The welcome screen check below needs to see `messages` as non-empty
#      so it can hide the example cards on the same render that begins
#      processing the new query.
# ----------------------------------------------------------------------------
typed_prompt = st.chat_input("Ask a question about OLED technology...")
prompt = typed_prompt or st.session_state.pop("pending_query", None)

# Commit the user message to history BEFORE rendering anything that depends
# on conversation state.
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

# Main Chat Interface
# Two columns: title on the left, "New chat" button on the right.
# Same pattern ChatGPT / Claude uses, so users don't have to hunt the sidebar.
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
            st.rerun()


# ----------------------------------------------------------------------------
# Welcome screen: example queries grouped by Strict-RAG decision tier.
#
# Why we show this:
# - Strict RAG is intentionally tuned for high-precision rejection of
#   off-topic queries. New users who don't yet know what the docs cover
#   can otherwise get only "OFF_TOPIC" rejections and leave confused.
# - Showing one-click examples for every tier (RAG / NO_ANSWER / OFF_TOPIC)
#   teaches the decision logic by demonstration in seconds.
#
# Implementation note:
# - Each example is a button. Clicking sets st.session_state.pending_query
#   and reruns. The chat-input block below treats pending_query exactly
#   like a typed prompt, so we don't duplicate the answer pipeline.
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
        "caption": (
            "Rejected at the planning step, before any search"
            if IS_AGENT
            else "Auto-rejected without calling the LLM"
        ),
        "queries": [
            "How do I bake a chocolate cake?",
            "Recommend me a Netflix show.",
        ],
    },
}

# ----------------------------------------------------------------------------
# Welcome screen container.
#
# Wrapping in st.empty() is intentional: when this slot is left untouched on
# a rerun (because show_welcome is False), Streamlit *guarantees* its DOM
# content is cleared. Without this wrapper, leftover example cards from the
# previous render can briefly stay visible during long-running answer
# generation while the spinner is up.
# ----------------------------------------------------------------------------
welcome_slot = st.empty()

# Welcome cards appear only when there's no conversation at all.
# Note: we already appended the user message above (if any), so an incoming
# prompt makes `messages` non-empty and naturally hides the welcome screen.
show_welcome = not st.session_state.messages

if show_welcome:
    with welcome_slot.container():
        st.markdown("### Try these example queries")
        if IS_AGENT:
            st.caption(
                "Click any example to see how the agent decides between three modes. "
                "Open the Agent Trace under each answer to see its plan and searches."
            )
        else:
            st.caption(
                "Click any example to see how Strict RAG decides between three modes. "
                "The relevance score determines whether the LLM is even called."
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
                        # Stash the query and rerun; the prompt resolver at
                        # the top of the script picks it up next render.
                        st.session_state.pending_query = query_text
                        st.rerun()
        st.markdown("---")

# Display Chat History
# This now also renders the brand-new user message we appended at the top,
# so we don't need a separate `with st.chat_message("user")` block below.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_message_extras(
            metadata=message.get("metadata"),
            sources=message.get("sources"),
            trace=message.get("trace"),
            sources_are_cited=message.get("sources_are_cited", False),
        )


def run_agent_with_progress(question):
    """Run the agent and show each plan and search step in a live status box."""
    with st.status("Planning...", expanded=False) as status_box:

        def on_event(event):
            # The agent calls this after every step, so the user can follow
            # the progress of a multi-search run instead of watching a silent
            # spinner.
            # NOTE: Events from parallel workers arrive in one batch when the
            # workers finish, because Streamlit can only be updated from the
            # main thread.
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

        result = assistant.query(question, on_event=on_event)
        status_box.update(label=f"Done ({result['agent']['stop_reason']})", state="complete")
    save_trace(result)
    return result


if prompt:
    # Generate the assistant response (user message was already added above).
    with st.chat_message("assistant"):
        start_time = time.time()

        if IS_AGENT:
            result = run_agent_with_progress(prompt)
        else:
            with st.spinner("Analyzing documents..."):
                result = assistant.query(prompt)
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
            mode_text = (
                "Off-Topic Rejection (Classified out of domain)"
                if IS_AGENT
                else "Off-Topic Rejection (Auto-Rejected, No LLM Call)"
            )
        elif mode == "ERROR":
            status_color = "gray"
            icon = "⚠️"
            mode_text = "Generation Error"
        else:
            status_color = "gray"
            icon = "⚪"
            mode_text = f"Unknown mode: {mode}"

        # `score` is the relevance of the single strongest retrieved document.
        status_text = f"{icon} **{mode_text}** | Top Relevance: {score:.3f}"
        if "agent" in result:
            searches = result["agent"]["searches"]
            # An out-of-domain question stops before any search, so showing
            # "Top Relevance: 0.000" would be misleading. We show the search
            # count instead.
            if searches == 0:
                status_text = f"{icon} **{mode_text}** | Searches: 0"
            else:
                status_text += f" | Searches: {searches}"
            
        # Display Answer
        message_placeholder.markdown(
            f"**Answer:** {answer}\n\n"
            f":{status_color}[{status_text}] | Time: {format_time(elapsed)}"
        )

        # Build the metadata + docs payload ONCE so the live render and the
        # saved-history entry stay in perfect sync.
        live_metadata = {
            "mode": mode,
            "response_time": f"{elapsed:.2f}s",
            "retrieval": result.get("retrieval_metadata", {}),
        }
        # We store plain dicts instead of Document objects so the chat history
        # stays light.
        if mode != "RAG":
            live_sources = None
        elif "sources" in result:
            live_sources = result["sources"]
        else:
            live_sources = sources_from_docs(result["retrieved_docs"])
        live_trace = result["agent"]["trace"] if "agent" in result else None

        # Render expanders right now so the user sees them immediately,
        # without waiting for the next Streamlit rerun.
        render_message_extras(
            metadata=live_metadata,
            sources=live_sources,
            trace=live_trace,
            sources_are_cited=IS_AGENT,
        )

        # Save to history so the same message keeps showing on later reruns.
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"**Answer:** {answer}\n\n:{status_color}[{status_text}] | Time: {format_time(elapsed)}",
            "metadata": live_metadata,
            "sources": live_sources,
            "trace": live_trace,
            "sources_are_cited": IS_AGENT,
        })
