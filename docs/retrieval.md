# Retrieval

> This page describes the retrieval stack in `src/retrieval.py`, the "R" in our RAG system. The agent reaches it only through its `search_documents` tool; see [Agent Engine](agent_engine.md) for how the agent plans, searches, and answers.

## Philosophy

For professional engineers at the PhD level, generic or hallucinated answers are unacceptable. Standard RAG systems often suffer from "General Knowledge Leakage," where the model fills gaps using non-technical internet data (news, blogs, Wikipedia).

Our **Strict RAG** design is meant to prevent this. Answers have to come from high-quality technical documents, and the agent may only cite chunks that retrieval actually returned.

### Key Principles

- **Documents as the Source of Truth**: The agent answers from the retrieved chunks and cites them.
- **No Hallucination**: It must never invent numbers, conditions, or citations that are not in the documents.
- **No Generic Fillers**: If the documents do not contain the answer, the agent returns `NO_ANSWER_IN_DOCS` instead of guessing from general training data.
- **Wide Retrieval, Compact Context**: Each search retrieves a wider candidate pool first, then filters and reranks it, so the agent only sees the strongest final chunks.

## The Relevance Score (Sigmoid)

In scientific domains, raw cosine similarity scores often cluster tightly (e.g., 0.75 vs 0.82), making it difficult to distinguish between "truly relevant" and "somewhat related."

We apply a **Sigmoid Transformation** to spread these scores out. This amplifies small differences in similarity, pushing ambiguous scores toward the extremes (0 or 1). This creates a sharper decision boundary, allowing for a clean and decisive cut-off.

$$ relevance = \frac{1}{1 + e^{-k(x - x_0)}} $$

- **$x$**: Cosine similarity of a **single** document against the query.
- **$x_0$ (Midpoint)**: 0.60. The cosine where relevance = 0.50, i.e. the document filter.
- **$k$ (Steepness)**: 17.27. Places relevance 0.25 at cosine 0.536.

### Distance → cosine (corrected)

Chroma's default `l2` space returns the **squared** Euclidean distance $d = \lVert q - e \rVert^2$. For the normalized BGE embeddings $d = 2 - 2\cos$, so

$$ \cos = 1 - \frac{d}{2} $$

Our earlier version treated $d$ as the plain (unsquared) distance and computed $1 - d^2/2$, which is not the cosine. We fixed the formula and refit the sigmoid at the same time ($x_0$ 0.68 → 0.60, $k$ 10 → 17.27), so that the calibrated boundaries (relevance 0.50 and 0.25) fall on exactly the same documents as before. The only difference is that the intermediate value is now a real cosine similarity.

### One score, one scale

Every threshold is compared against **relevance**, never against raw cosine similarity. Raw values exist only as the input to the sigmoid, inside `Retriever.distance_to_relevance()`.

The midpoint and steepness are the sole exception: they are parameters _of_ the transform rather than decision thresholds, so by definition they live in raw space and cannot be restated in relevance space.

Scoring each document individually (rather than averaging a fixed top-4) matters because averaging lets a handful of weak chunks drown out one strong match.

## Retrieval Pipeline

Every `search_documents(query)` call runs these steps:

1. **Wide vector search**: ChromaDB retrieves `CANDIDATE_TOP_K = 20` candidates using BGE embeddings. Chroma ranks by L2 distance internally, which costs us nothing: the sigmoid is monotonic in similarity and similarity is monotonic in distance, so the top-k by distance is exactly the top-k by relevance.
2. **Relevance conversion**: every candidate is scored on the relevance scale.
3. **Document filtering**: candidates with `relevance >= MIN_DOC_RELEVANCE` (0.50) survive.
4. **BGE reranking**: if more than `FINAL_TOP_N = 4` documents survive, `BAAI/bge-reranker-base` rescores each query-document pair and keeps the strongest 4.
5. **Return to the agent**: the final chunks get stable IDs and go into the request's evidence ledger. If nothing survives, the tool returns `no_relevant_documents` together with the best relevance it saw, and the agent decides whether a reworded search is worth trying.

This improves recall compared with fixed top-4 retrieval while keeping each tool result compact and high precision.

## What the agent can and cannot change

| Decided by the agent | Fixed in Python (config) |
| :-- | :-- |
| The query text, and whether to search again | `CANDIDATE_TOP_K`, `MIN_DOC_RELEVANCE`, `FINAL_TOP_N`, the reranker, the sigmoid |
| Which returned chunks to cite | Which chunks may be cited (only those returned in this request) |

The relevance score is also used inside the agent in one place: a light-model "insufficient evidence" result escalates to the heavy model only when retrieval scored at least `AGENT_ESCALATE_MIN_RELEVANCE` (0.70), because then the evidence was probably there.
