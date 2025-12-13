"""
AI-Driven OLED Assistant - Streamlit Application
Aligned with OLED_assistant_v3_final.ipynb
"""
import streamlit as st
import time
from rag_engine import StrictRAGAssistant, create_embeddings, get_vectorstore
import config
from utils import logger, format_time

# Page Configuration
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide"
)

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

# Main Chat Interface
st.title(config.APP_TITLE)
st.markdown("#### Intelligent Engineering Support System")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "metadata" in message and message["metadata"]:
            with st.expander("Analysis Details"):
                st.json(message["metadata"])
        if "docs" in message and message["docs"]:
            with st.expander("Retrieved Documents"):
                for i, doc in enumerate(message["docs"], 1):
                    st.markdown(f"**Doc {i}**")
                    st.text(doc.page_content[:300] + "...")

# Chat Input
if prompt := st.chat_input("Ask a question about OLED technology..."):
    # User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant Response
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
        
        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"**Answer:** {answer}\n\n:{status_color}[{status_text}] | Time: {format_time(elapsed)}",
            "metadata": {
                "mode": mode,
                "relevance_score": score,
                "response_time": f"{elapsed:.2f}s"
            },
            "docs": result["retrieved_docs"] if mode == "RAG" else None
        })
