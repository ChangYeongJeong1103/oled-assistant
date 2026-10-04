import os
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "chroma_db")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
DOCS_FOLDER = os.path.join(BASE_DIR, "data")

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
#
# NOTE: Chroma's default "l2" space returns the SQUARED Euclidean distance, so
# for unit vectors cosine = 1 - distance / 2 (see
# Retriever.distance_to_relevance). The midpoint is the cosine where
# relevance = 0.5, which is our document filter. The steepness puts relevance
# 0.25 at cosine 0.536. Both values were calibrated on this corpus
# (docs/hyperparameter.md).
SIGMOID_MIDPOINT = 0.60
SIGMOID_STEEPNESS = 17.27

# Retrieval Settings
# Step 1: retrieve a wider candidate pool from ChromaDB.
# Step 2: keep documents whose relevance clears MIN_DOC_RELEVANCE.
# Step 3: rerank the survivors and return only FINAL_TOP_N to the agent.
CANDIDATE_TOP_K = int(os.getenv("CANDIDATE_TOP_K", "20"))
FINAL_TOP_N = int(os.getenv("FINAL_TOP_N", "4"))
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "BAAI/bge-reranker-base",
)

# Document filter (relevance space, i.e. post-sigmoid)
#
# MIN_DOC_RELEVANCE decides which documents are good enough to rerank and show
# to the agent. It is the knob for context quality. Measured on the current
# corpus: on-topic queries keep 7-20 of 20 candidates here, which leaves the
# cross-encoder something to choose from, while off-topic queries keep none.
MIN_DOC_RELEVANCE = float(os.getenv("MIN_DOC_RELEVANCE", "0.50"))


def _env_flag(name, default):
    """Read a true/false flag from an environment variable."""
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Agent models (OpenAI Responses API)
# ---------------------------------------------------------------------------
# When routing is off, AGENT_MODEL is used for every role. When routing is on,
# the planner and easy questions use the LIGHT model, while hard questions
# (and escalations) use the HEAVY model.
AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-6-luna")
AGENT_LIGHT_MODEL = os.getenv("AGENT_LIGHT_MODEL", "gpt-6-luna")
AGENT_HEAVY_MODEL = os.getenv("AGENT_HEAVY_MODEL", "gpt-6.1-sol")

# Reasoning effort we send for each model. The allowed values differ by model
# family, e.g. gpt-5 accepts minimal/low/..., gpt-6-luna accepts none/low/...,
# and gpt-6.1-sol accepts low/...
# If AGENT_REASONING_EFFORT is set, it overrides this table for every model.
MODEL_REASONING_EFFORT = {
    "gpt-6-luna": "low",
    "gpt-6.1-sol": "low",
    "gpt-5-mini": "minimal",
    "gpt-5": "minimal",
}
AGENT_REASONING_EFFORT = os.getenv("AGENT_REASONING_EFFORT", "")


def reasoning_effort_for(model_name):
    """Return the reasoning effort to send for a model, or None to leave the parameter out."""
    return AGENT_REASONING_EFFORT or MODEL_REASONING_EFFORT.get(model_name)


# ---------------------------------------------------------------------------
# Agent features (each can be switched on/off for evaluation)
# ---------------------------------------------------------------------------
# The defaults below come from our evaluation (docs/agent_engine.md).
# Repeated tests during development showed that the workers and the reviewer
# solve different problems: workers made sequential retrieval more reliable,
# while the reviewer consistently rejected answers that only covered facts
# related to the question. The combined setup's final 50-question run had 0%
# false acceptance and 0.35% unsupported claims, so both are enabled by default.
#
# Parallel tool calling: the model may issue several independent searches in
# one turn.
AGENT_PARALLEL_SEARCH = _env_flag("AGENT_PARALLEL_SEARCH", "true")
# Model routing: the planner labels each question's complexity and hard
# questions go to the heavy model. If a light-model run fails, it may escalate
# to the heavy model once.
AGENT_ROUTING = _env_flag("AGENT_ROUTING", "true")
# Orchestrator-worker: number of research workers (0 = single agent).
AGENT_WORKERS = int(os.getenv("AGENT_WORKERS", "2"))
# Reviewer: once an answer's citations are valid, the reviewer checks each of
# its claims against the cited chunks before we accept the answer.
AGENT_REVIEWER = _env_flag("AGENT_REVIEWER", "true")
# We only escalate a light-model "insufficient evidence" result when retrieval
# looked strong, because then the evidence was probably there and the light
# model misread it.
AGENT_ESCALATE_MIN_RELEVANCE = float(os.getenv("AGENT_ESCALATE_MIN_RELEVANCE", "0.70"))

# ---------------------------------------------------------------------------
# Agent budgets
# ---------------------------------------------------------------------------
# IMPORTANT: All of these limits are enforced in our Python code instead of
# being left to the prompt.
#
# Each user question gets one shared budget, and every LLM request counts
# against it (planning, research turns, corrections, workers, orchestrator,
# reviewer, escalation). The default call budget grows with the features that
# need extra calls: 6 to start, +4 with routing, +6 with workers, and +3 with
# the reviewer.
_default_llm_calls = (
    6
    + (4 if AGENT_ROUTING else 0)
    + (6 if AGENT_WORKERS else 0)
    + (3 if AGENT_REVIEWER else 0)
)
AGENT_MAX_LLM_CALLS = int(os.getenv("AGENT_MAX_LLM_CALLS", str(_default_llm_calls)))
AGENT_MAX_SEARCHES = int(os.getenv("AGENT_MAX_SEARCHES", "4" if AGENT_WORKERS else "3"))
# How many times an answer may be resubmitted after failing validation, and
# how many reviewer rounds may send it back.
AGENT_MAX_ANSWER_REVISIONS = int(os.getenv("AGENT_MAX_ANSWER_REVISIONS", "2"))
AGENT_MAX_REVIEW_ROUNDS = int(os.getenv("AGENT_MAX_REVIEW_ROUNDS", "2"))
# Maximum LLM calls for one research loop (the single agent, one worker, or
# the escalated heavy run) before it has to stop. This keeps one loop from
# using up the calls the other steps still need.
AGENT_MAX_CALLS_PER_LOOP = int(os.getenv("AGENT_MAX_CALLS_PER_LOOP", "5"))
# AGENT_DEADLINE_SECONDS is the wall-clock budget for one user question
# (150s when workers, the reviewer, or routing is on, otherwise 90s).
# AGENT_API_TIMEOUT_SECONDS is the timeout for a single API attempt, and it is
# always capped by the time left before the question's deadline.
_default_deadline = 150 if (AGENT_WORKERS or AGENT_REVIEWER or AGENT_ROUTING) else 90
AGENT_DEADLINE_SECONDS = float(os.getenv("AGENT_DEADLINE_SECONDS", str(_default_deadline)))
AGENT_API_TIMEOUT_SECONDS = float(os.getenv("AGENT_API_TIMEOUT_SECONDS", "60"))
# Retries for transient API errors (timeouts, connection errors, 429, 5xx).
# Our runtime does the retrying instead of the SDK, so that every attempt
# still fits inside the deadline.
AGENT_API_RETRIES = int(os.getenv("AGENT_API_RETRIES", "1"))
# Size limits on the tool arguments the model produces.
AGENT_MAX_QUERY_CHARS = 300
AGENT_MAX_ANSWER_CHARS = 6000
AGENT_MAX_SUBQUESTIONS = 3

# The app appends one JSON line per agent run to this file (best effort).
AGENT_TRACE_PATH = os.path.join(LOGS_DIR, "agent_traces.jsonl")

# Paper title and URL for each source file name (see source_registry.py).
SOURCE_REGISTRY_PATH = os.path.join(BASE_DIR, "src", "source_registry.json")

# ---------------------------------------------------------------------------
# Token pricing (USD per 1M tokens) used for cost reporting only.
# TODO: Re-check against https://openai.com/api/pricing when models change.
# ---------------------------------------------------------------------------
MODEL_PRICING_PER_1M = {
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    # Standard short-context rates (prompts up to 272K input tokens), taken
    # from https://developers.openai.com/api/docs/pricing (checked 2026-10-03).
    # NOTE: We don't model the cache-write charge (1.25x the input rate).
    "gpt-6-luna": {"input": 0.10, "cached_input": 0.01, "output": 0.50},
    "gpt-6.1-sol": {"input": 2.00, "cached_input": 0.10, "output": 10.00},
}

# UI Settings
APP_TITLE = "AI-Driven OLED Assistant"
APP_ICON = "⚛"  # Atom symbol - fits OLED/physics theme
