import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "chroma_db")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# LLM Settings: Mistral-Nemo 12B (Ollama local server)
# NOTE: Aligned with OLED_assistant_v3_final.ipynb
LLM_MODEL = "mistral-nemo"
LLM_TEMPERATURE = 0.2

# Ollama OpenAI-compatible endpoint
LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
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
SIGMOID_MIDPOINT = 0.50
SIGMOID_STEEPNESS = 18

# UI Settings
APP_TITLE = "AI-Driven OLED Assistant"
APP_ICON = "⚛"  # Atom symbol - fits OLED/physics theme
