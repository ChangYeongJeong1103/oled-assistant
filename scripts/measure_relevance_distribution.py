"""
Measure the relevance score distribution produced by the retrieval stage.

Why this script exists
----------------------
Every search the agent runs goes through the document filter
(MIN_DOC_RELEVANCE): only documents above it reach the reranker and the agent.

The threshold must be expressed as RELEVANCE = sigmoid(cosine similarity),
never as raw cosine similarity. Raw scores in scientific corpora cluster very
tightly (0.75 vs 0.82), which makes them useless as a decision axis.

This script runs a set of representative queries against the existing ChromaDB
and prints the relevance distribution for every retrieved candidate, so the
threshold can be chosen from measured data instead of guesswork. It also shows
how far on-topic and off-topic queries are apart.

The script is READ-ONLY: it never rebuilds or writes to the vector store.

Usage
-----
    python scripts/measure_relevance_distribution.py
"""

import math
import os
import sys

# Make `src/` importable so we can reuse the project's own config values.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from langchain_community.embeddings import HuggingFaceEmbeddings  # noqa: E402
from langchain_community.vectorstores import Chroma  # noqa: E402

import config  # noqa: E402


# ---------------------------------------------------------------------------
# Representative queries, grouped by the answer mode we EXPECT.
#
# These mirror the example queries shown on the app's welcome screen, plus a
# few "narrow" questions. Narrow questions are the interesting case: they are
# clearly on-topic but usually match only one or two strong chunks, which is
# exactly the situation where a top-4 average unfairly rejects them.
# ---------------------------------------------------------------------------
QUERY_GROUPS = {
    "ON_TOPIC (broad)": [
        "What is OLED operation principle?",
        "What is the role of the hole transport layer in OLED devices?",
        "What are typical electron mobility values in OLED ETL materials?",
    ],
    "ON_TOPIC (narrow)": [
        "How does thermally activated delayed fluorescence work?",
        "What causes efficiency roll-off at high current density?",
        "Explain the outcoupling efficiency limit of bottom-emission OLEDs.",
    ],
    "ON_TOPIC (docs may not cover)": [
        "What is the supply chain cost of phosphorescent OLEDs?",
        "Which company filed the most OLED patents last year?",
    ],
    "OFF_TOPIC": [
        "How do I bake a chocolate cake?",
        "Recommend me a Netflix show.",
        "What is the capital city of France?",
    ],
}


def distance_to_similarity(distance):
    """
    Convert a ChromaDB L2 distance into a raw cosine similarity.

    NOTE: Chroma's "l2" space returns the SQUARED Euclidean distance. The
    BGE embeddings are stored normalized, so for unit vectors:
        d = ||q - e||^2 = 2 - 2*cos  ->  cos = 1 - d / 2
    The result is clamped to [0, 1] to absorb small numerical drift.
    """
    similarity = 1.0 - float(distance) / 2.0
    return max(0.0, min(1.0, similarity))


def similarity_to_relevance(similarity):
    """
    Apply the sigmoid transformation that turns similarity into RELEVANCE.

    This is the only score the pipeline should ever compare against a
    threshold. The midpoint/steepness come from config so this script always
    reflects the app's live settings.
    """
    exponent = -config.SIGMOID_STEEPNESS * (similarity - config.SIGMOID_MIDPOINT)
    return 1.0 / (1.0 + math.exp(exponent))


def load_vectorstore():
    """
    Open the persisted ChromaDB read-only.

    We deliberately do NOT call get_or_create_vectorstore(): that helper
    rebuilds from `data/` when the DB looks incompatible, and the public repo
    has no `data/` folder. Failing loudly is safer for a measurement run.
    """
    if not os.path.isdir(config.DB_PATH) or not os.listdir(config.DB_PATH):
        raise SystemExit(f"No ChromaDB found at {config.DB_PATH}")

    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        persist_directory=config.DB_PATH,
        embedding_function=embeddings,
    )
    print(f"Loaded ChromaDB with {vectorstore._collection.count()} chunks.\n")
    return vectorstore


def score_query(vectorstore, query):
    """
    Retrieve the candidate pool for one query and return relevance scores.

    Returns a list of relevance values sorted high -> low, one per candidate.
    """
    hits = vectorstore.similarity_search_with_score(query, k=config.CANDIDATE_TOP_K)

    relevances = []
    for _, distance in hits:
        similarity = distance_to_similarity(distance)
        relevances.append(similarity_to_relevance(similarity))

    # Chroma returns nearest-first, but sort explicitly so the statistics below
    # never depend on backend ordering.
    relevances.sort(reverse=True)
    return relevances


def summarize(relevances):
    """Compute the candidate gate statistics we want to compare."""
    if not relevances:
        return {"max": 0.0, "top2_mean": 0.0, "top4_mean": 0.0}

    top2 = relevances[:2]
    top4 = relevances[: config.FINAL_TOP_N]
    return {
        "max": relevances[0],
        "top2_mean": sum(top2) / len(top2),
        "top4_mean": sum(top4) / len(top4),
    }


def main():
    vectorstore = load_vectorstore()

    # Collected per group so we can look for a clean on-topic / off-topic split.
    results = []

    for group_name, queries in QUERY_GROUPS.items():
        print("=" * 78)
        print(group_name)
        print("=" * 78)

        for query in queries:
            relevances = score_query(vectorstore, query)
            stats = summarize(relevances)
            results.append((group_name, query, stats, relevances))

            print(f"\n  Q: {query}")
            print(
                f"     max={stats['max']:.3f}   "
                f"top2_mean={stats['top2_mean']:.3f}   "
                f"top4_mean={stats['top4_mean']:.3f}"
            )
            # Show the full pool so we can see where relevance falls off a cliff.
            pool = "  ".join(f"{value:.3f}" for value in relevances)
            print(f"     pool: {pool}")

            # How many documents would survive various filter thresholds?
            counts = [
                f"{threshold:.2f}->{sum(1 for v in relevances if v >= threshold)}"
                for threshold in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60)
            ]
            print(f"     survivors by filter threshold: {'  '.join(counts)}")

            # What the agent's search tool would return with the live config.
            print(f"     -> search tool returns: {predict_search_status(relevances)}")
        print()

    print_separation_report(results)


def predict_search_status(relevances):
    """
    Replay what search_documents returns using the thresholds in config.

    Kept in sync with Retriever + SearchTool: documents above the filter are
    reranked when there are more than FINAL_TOP_N of them. With no survivors,
    the agent gets "no_relevant_documents" and decides whether to search again.
    """
    survivors = [value for value in relevances if value >= config.MIN_DOC_RELEVANCE]
    if not survivors:
        return "no_relevant_documents"
    # The reranker is skipped when the pool is already small enough.
    reranked = "reranked" if len(survivors) > config.FINAL_TOP_N else "no rerank"
    return f"ok ({len(survivors)} survivors, {reranked})"


def print_separation_report(results):
    """
    Show how well each statistic separates on-topic from off-topic queries.

    A statistic separates well when the WORST on-topic query still scores
    higher than the BEST off-topic query. The gap between those two numbers is
    the safety margin for a threshold.
    """
    print("=" * 78)
    print("ON-TOPIC / OFF-TOPIC SEPARATION")
    print("=" * 78)

    on_topic = [stats for group, _, stats, _ in results if not group.startswith("OFF_TOPIC")]
    off_topic = [stats for group, _, stats, _ in results if group.startswith("OFF_TOPIC")]

    for key in ("max", "top2_mean", "top4_mean"):
        worst_on = min(stats[key] for stats in on_topic)
        best_off = max(stats[key] for stats in off_topic)
        margin = worst_on - best_off
        verdict = "OK" if margin > 0 else "OVERLAP"
        print(
            f"  {key:10s}  worst on-topic={worst_on:.3f}   "
            f"best off-topic={best_off:.3f}   margin={margin:+.3f}  [{verdict}]"
        )


if __name__ == "__main__":
    main()
