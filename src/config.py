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

# Retrieval Settings
# Step 1: retrieve a wider candidate pool from ChromaDB.
# Step 2: keep only documents with enough embedding similarity.
# Step 3: rerank the surviving documents and send only FINAL_TOP_N to the LLM.
CANDIDATE_TOP_K = int(os.getenv("CANDIDATE_TOP_K", "20"))
MIN_DOCUMENT_SIMILARITY = float(os.getenv("MIN_DOCUMENT_SIMILARITY", "0.50"))
FINAL_TOP_N = int(os.getenv("FINAL_TOP_N", "4"))
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "BAAI/bge-reranker-base",
)

# Legacy alias used by older notebooks/docs. The app now uses FINAL_TOP_N.
TOP_K_DOCUMENTS = FINAL_TOP_N

# Strict RAG Thresholds
RELEVANCE_THRESHOLD = 0.60
SIGMOID_MIDPOINT = 0.68
SIGMOID_STEEPNESS = 10

# UI Settings
APP_TITLE = "AI-Driven OLED Assistant"
APP_ICON = "⚛"  # Atom symbol - fits OLED/physics theme
