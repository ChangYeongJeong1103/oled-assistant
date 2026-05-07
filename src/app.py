"""
AI-Driven OLED Assistant - Streamlit Application
Aligned with OLED_assistant_v3_final.ipynb
"""
import streamlit as st
import time
import os
from rag_engine import StrictRAGAssistant, create_embeddings, get_vectorstore
import config
from utils import logger, format_time

# Page Configuration
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide"
)


def format_doc_source(doc):
    """
    Convert a retrieved Document's metadata into a clean human-readable label.

    Why we need this:
    - PyPDFLoader sets doc.metadata = {"source": "/full/path/to.pdf", "page": 0}
    - Docx2txtLoader sets doc.metadata = {"source": "/full/path/to.docx"}
    - We only want the file *name* (engineers don't care about /app/data/ prefix)
      and we want page numbers to be 1-indexed for human readability.
    """
    # Default to empty string so .get(...) never returns None for basename()
    source_path = doc.metadata.get("source", "")
    if not source_path:
        return "Unknown source"

    # Strip directory prefix so we only show e.g. "oled_book.pdf"
    file_name = os.path.basename(source_path)

    # PyPDFLoader exposes 0-indexed page numbers; convert to 1-indexed.
    page = doc.metadata.get("page")
    if page is not None:
        return f"{file_name} (p.{page + 1})"
    return file_name


def render_message_extras(metadata=None, docs=None):
    """
    Render the "Analysis Details" and "Retrieved Documents" expanders.

    Why we need this helper:
    Streamlit redraws each chat message in TWO different code paths:
      1) the chat-history loop (runs on every page rerun)
      2) the chat-input block (runs ONCE, right after the user hits send)
    If the expanders only exist in path #1, the brand-new message a user just
    sent will look bare until the page reruns again. Centralising the render
    code here keeps both paths identical and avoids that "missing expander"
    bug on the very first render.
    """
    if metadata:
        with st.expander("Analysis Details"):
            st.json(metadata)
    if docs:
        with st.expander("Retrieved Documents"):
            # Show only the source label (file name + page) so engineers can
            # verify provenance at a glance without wading through raw chunks.
            for i, doc in enumerate(docs, 1):
                source_label = format_doc_source(doc)
                st.markdown(f"**Doc {i}** — `{source_label}`")

# Initialize Assistant (Cached to prevent reloading on every interaction)
@st.cache_resource
def get_assistant():
    embeddings = create_embeddings()
    vectorstore = get_vectorstore(embeddings)
    return StrictRAGAssistant(
        vectorstore=vectorstore,
        llm_model=config.LLM_MODEL,
        relevance_threshold=config.RELEVANCE_THRESHOLD,
        top_k=config.TOP_K_DOCUMENTS,
        temperature=config.LLM_TEMPERATURE,
        sigmoid_midpoint=config.SIGMOID_MIDPOINT,
        sigmoid_steepness=config.SIGMOID_STEEPNESS,
    )

try:
    if not os.environ.get("OPENAI_API_KEY"):
        st.error(
            "❌ OPENAI_API_KEY not found. Set it in your environment or `.env` file before running the app."
        )
        st.stop()
    assistant = get_assistant()
except Exception as e:
    st.error(f"Failed to initialize RAG Engine: {str(e)}")
    st.stop()

# Sidebar
with st.sidebar:
    st.title(f"{config.APP_ICON} OLED Assistant")
    st.markdown("---")
    st.markdown("**System Status**")
    st.success(f"Model: {config.LLM_MODEL}")
    st.info(f"Strict Threshold = {config.RELEVANCE_THRESHOLD}")
    st.markdown("---")
    st.markdown("### User Guide")
    st.markdown("""
    1. Ask questions about **OLED physics, materials, or fabrication**.
    2. The system checks **relevance** first.
    3. If relevant, it uses documents as the **PRIMARY** source and may fill in missing logical steps.
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
        "caption": "Auto-rejected without calling the LLM",
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
            docs=message.get("docs"),
        )

if prompt:
    # Generate the assistant response (user message was already added above).
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        start_time = time.time()
        
        with st.spinner("Analyzing documents..."):
            result = assistant.query(prompt)
            
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
            mode_text = "Off-Topic Rejection (Low Relevance, Auto-Rejected)"
        else:
            status_color = "gray"
            icon = "⚪"
            mode_text = f"Unknown mode: {mode}"

        status_text = f"{icon} **{mode_text}** | Relevance Score: {score:.3f}"
            
        # Display Answer
        message_placeholder.markdown(
            f"**Answer:** {answer}\n\n"
            f":{status_color}[{status_text}] | Time: {format_time(elapsed)}"
        )

        # Build the metadata + docs payload ONCE so the live render and the
        # saved-history entry stay in perfect sync.
        live_metadata = {
            "mode": mode,
            "relevance_score": score,
            "response_time": f"{elapsed:.2f}s",
        }
        live_docs = result["retrieved_docs"] if mode == "RAG" else None

        # Render expanders right now so the user sees them immediately,
        # without waiting for the next Streamlit rerun.
        render_message_extras(metadata=live_metadata, docs=live_docs)

        # Save to history so the same message keeps showing on later reruns.
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"**Answer:** {answer}\n\n:{status_color}[{status_text}] | Time: {format_time(elapsed)}",
            "metadata": live_metadata,
            "docs": live_docs,
        })
