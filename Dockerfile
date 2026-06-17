FROM python:3.11-slim

# ================================
# Runtime environment
# ================================
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Optional build tools for Python packages that may need compilation.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# ================================
# Python dependencies
# ================================
# Copy requirements first for better Docker layer caching.
COPY requirements.txt ./requirements.txt

# Cloud Run is CPU-based here. Preinstall the CPU-only PyTorch wheel before
# sentence-transformers so pip does not pull large CUDA/NVIDIA dependencies.
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Pre-download model weights during the image build so Cloud Run does not
# spend cold-start time downloading them after a user opens the live demo.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-m3'); CrossEncoder('BAAI/bge-reranker-base')"

# ================================
# Application files
# ================================
COPY src/ ./src
COPY docs/ ./docs
COPY chroma_db/ ./chroma_db

# Streamlit default port
EXPOSE 8501

# ================================
# Start app
# ================================
# OPENAI_API_KEY must be provided at runtime.
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
