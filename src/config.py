import os
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "chroma_db")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
DOCS_FOLDER = os.path.join(BASE_DIR, "data")

# LLM Settings: Cloud demo mode (OpenAI API)
# Default model is GPT-5-mini (OpenAI API).
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5-mini")
# NOTE: GPT-5 family models (e.g., gpt-5-mini) only support the default
# temperature (1.0). The value below is still used for older models such as
# gpt-4o-mini; rag_engine.create_llm() decides whether to actually send it.
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
# Embedding Settings
# IMPORTANT: Must match the embedding model used to build the persisted ChromaDB.
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_BATCH_SIZE = 16

# RAG Settings
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 500

# ---------------------------------------------------------------------------
# Relevance scale
# ---------------------------------------------------------------------------
# Every threshold in this project is compared against RELEVANCE, defined as:
#
#     relevance = sigmoid(cosine_similarity)
#
# Raw cosine similarity is NEVER compared against a threshold. In scientific
# corpora raw scores cluster very tightly (0.75 vs 0.82), which makes them a
# poor decision axis. The sigmoid spreads that narrow band into a usable range.
#
# The two constants below are the ONLY values that live in raw-similarity
# space, because they are parameters *of* the transform rather than decision
# thresholds. They cannot be expressed in relevance space by definition.
SIGMOID_MIDPOINT = 0.68
SIGMOID_STEEPNESS = 10

# Retrieval Settings
# Step 1: retrieve a wider candidate pool from ChromaDB.
# Step 2: keep documents whose relevance clears MIN_DOC_RELEVANCE.
# Step 3: rerank the survivors and send only FINAL_TOP_N to the LLM.
CANDIDATE_TOP_K = int(os.getenv("CANDIDATE_TOP_K", "20"))
FINAL_TOP_N = int(os.getenv("FINAL_TOP_N", "4"))
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "BAAI/bge-reranker-base",
)

# Strict RAG Thresholds (relevance space, i.e. post-sigmoid)
#
# MIN_DOC_RELEVANCE decides which documents are good enough to rerank and show
# to the LLM. It is the knob for context quality. Measured on the current
# corpus: on-topic queries keep 7-20 of 20 candidates here, which leaves the
# cross-encoder something to choose from, while off-topic queries keep none.
MIN_DOC_RELEVANCE = float(os.getenv("MIN_DOC_RELEVANCE", "0.50"))

# OFF_TOPIC_THRESHOLD only decides WHICH rejection the user sees when no
# document survives the filter: a genuinely out-of-domain question, or an
# in-domain question our documents happen not to cover. It never controls
# which documents reach the LLM, so it is deliberately loose. Measured on the
# current corpus: off-topic queries peak at 0.044, on-topic at 0.635+.
OFF_TOPIC_THRESHOLD = float(os.getenv("OFF_TOPIC_THRESHOLD", "0.25"))

# UI Settings
APP_TITLE = "AI-Driven OLED Assistant"
APP_ICON = "⚛"  # Atom symbol - fits OLED/physics theme
