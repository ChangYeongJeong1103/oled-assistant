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
TOP_K_DOCUMENTS = 4

# Strict RAG Thresholds
RELEVANCE_THRESHOLD = 0.60
SIGMOID_MIDPOINT = 0.68
SIGMOID_STEEPNESS = 10

# UI Settings
APP_TITLE = "AI-Driven OLED Assistant"
APP_ICON = "⚛"  # Atom symbol - fits OLED/physics theme
