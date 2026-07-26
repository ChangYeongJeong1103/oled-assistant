# Strict RAG Engine

## Philosophy
For professional engineers at the PhD level, generic or hallucinated answers are unacceptable. Standard RAG systems often suffer from "General Knowledge Leakage," where the model fills gaps using non-technical internet data (news, blogs, Wikipedia).

Our **Strict RAG** architecture is designed to prevent this. It ensures answers are derived **exclusively** from high-quality internal technical documents, providing insights that cannot be found via simple Google searches.

### Key Principles
- **Internal Documents as the Absolute Truth**: The model must prioritize the provided context above all else.
- **No Hallucination**: It must never invent numbers, conditions, or citations not present in the documents.
- **Zero Tolerance for Generic Fillers**: If the internal documents do not contain the answer, the model must report that through its `answer_found` flag rather than guessing based on general training data.
- **Wide Retrieval, Compact Context**: The system retrieves a wider candidate pool first, then filters and reranks candidates so the LLM only sees the strongest final documents.

## The Relevance Score (Sigmoid)

In scientific domains, raw cosine similarity scores often cluster tightly (e.g., 0.75 vs 0.82), making it difficult to distinguish between "truly relevant" and "somewhat related."

We apply a **Sigmoid Transformation** to spread these scores out. This amplifies small differences in similarity, pushing ambiguous scores toward the extremes (0 or 1). This creates a sharper decision boundary, allowing for a clean and decisive cut-off.

$$ relevance = \frac{1}{1 + e^{-k(x - x_0)}} $$

- **$x$**: Cosine similarity of a **single** document against the query.
- **$x_0$ (Midpoint)**: 0.68. The center of the current decision boundary.
- **$k$ (Steepness)**: 10. Controls how aggressively we separate "relevant" from "irrelevant".

### One score, one scale

Every threshold in the engine is compared against **relevance**, never against
raw cosine similarity. Raw values exist only as the input to the sigmoid, inside
`StrictRAGAssistant.distance_to_relevance()`.

The midpoint and steepness are the sole exception: they are parameters *of* the
transform rather than decision thresholds, so by definition they live in raw
space and cannot be restated in relevance space.

Scoring each document individually (rather than averaging a fixed top-4) matters
because averaging lets a handful of weak chunks drown out one strong match. A
narrow but perfectly valid question would otherwise be rejected as off-topic.

## Retrieval Pipeline

1. **Wide vector search**: ChromaDB retrieves `CANDIDATE_TOP_K = 20` candidates using BGE embeddings.
   Chroma ranks by L2 distance internally, which costs us nothing: the sigmoid is
   monotonic in similarity and similarity is monotonic in distance, so the top-k
   by distance is exactly the top-k by relevance.
2. **Relevance conversion**: every candidate is scored on the relevance scale.
3. **Document filtering**: candidates with `relevance >= MIN_DOC_RELEVANCE` survive.
4. **BGE reranking**: if more than `FINAL_TOP_N = 4` documents survive, `BAAI/bge-reranker-base` rescores each query-document pair and keeps the strongest 4.
5. **Generation**: the final documents become the LLM context.

This improves recall compared with fixed top-4 retrieval while keeping the final LLM context compact and high precision.

## 3-Tier Decision Logic

The engine classifies a query into one of three modes. Note that the document
filter, not the off-topic gate, is what decides whether an answer is possible.

### 1. 🟢 RAG Mode (Answered)
- **Condition**: At least one document clears `MIN_DOC_RELEVANCE = 0.50`, and the LLM reports `answer_found: true`.
- **Behavior**: The system generates a technical answer grounded in the reranked documents.

### 2. 🟠 No Answer Mode (Low Context)
- **Condition**: Either no document clears the filter while the best one still scores at or above `OFF_TOPIC_THRESHOLD = 0.25`, or documents were supplied but the LLM reports `answer_found: false`.
- **Behavior**: The system states that the documents do not cover the question.
- **Why**: To serve PhD-level experts, we avoid low-quality answers based on general internet data (blogs/Wikipedia). If our internal high-quality data cannot answer it, we prefer "No Answer" over a potentially misleading or generic guess.

### 3. 🔴 Off-Topic Mode (Rejected)
- **Condition**: No document clears the filter **and** the best document scores below `OFF_TOPIC_THRESHOLD = 0.25`.
- **Behavior**: The system rejects the query immediately without calling the LLM generation step.
- **Example**: "How to bake a cake?" or "Recommend me a Netflix show."

### Why the gate is loose and the filter is strict

The two thresholds answer different questions:

| | Question it answers | What it controls |
| :--- | :--- | :--- |
| `MIN_DOC_RELEVANCE = 0.50` | Is this document worth showing the LLM? | Answer quality |
| `OFF_TOPIC_THRESHOLD = 0.25` | Was this question ever about OLED? | Which rejection message appears |

Poor documents can never reach the LLM regardless of the gate, because the
filter removes them first. The gate is therefore free to be generous, which
protects legitimate but oddly phrased questions from being rejected outright.

## Detecting "No Answer" Without a Second LLM Call

The prompt asks the model to reply with a JSON envelope:

```json
{"answer_found": true, "answer": "..."}
```

Reading an explicit flag replaces the earlier approach of searching the response
for the phrase "Information not found", which silently failed whenever the model
reworded its refusal ("I couldn't find...", "The documents omit...").

An LLM-as-a-judge scoring pass was considered and rejected: it would double the
number of LLM calls per query, and it grades *answer quality* rather than the
thing Strict RAG actually needs to know, namely whether the answer was grounded
in the documents. The `answer_found` flag rides along with the answer the model
is already generating, so latency is unchanged.

If the JSON cannot be parsed, the engine treats the whole response as a normal
answer. A formatting slip should degrade to "answered" rather than discard a
good answer.
