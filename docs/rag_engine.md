# Strict RAG Engine

## Philosophy
For professional engineers at the PhD level, generic or hallucinated answers are unacceptable. Standard RAG systems often suffer from "General Knowledge Leakage," where the model fills gaps using non-technical internet data (news, blogs, Wikipedia).

Our **Strict RAG** architecture is designed to prevent this. It ensures answers are derived **exclusively** from high-quality internal technical documents, providing insights that cannot be found via simple Google searches.

### Key Principles
- **Internal Documents as the Absolute Truth**: The model must prioritize the provided context above all else.
- **No Hallucination**: It must never invent numbers, conditions, or citations not present in the documents.
- **Zero Tolerance for Generic Fillers**: If the internal documents do not contain the answer, the system must explicitly state "Information not found" rather than guessing based on general training data.

## 3-Tier Decision Logic

The engine classifies query into one of three modes:

### 1. 🟢 RAG Mode (Answered)
- **Condition**: Relevance Score ≥ Threshold (0.60) AND Documents contain the answer.
- **Behavior**: The system generates a technical answer citing the retrieved documents.

### 2. 🟠 No Answer Mode (Low Context)
- **Condition**: Relevance Score ≥ Threshold (0.60) BUT Documents do *not* contain the specific answer.
- **Behavior**: The LLM explicitly states "Information not found in the provided OLED documents."
- **Why**: To serve PhD-level experts, we avoid low-quality answers based on general internet data (blogs/Wikipedia). If our internal high-quality data cannot answer it, we prefer "No Answer" over a potentially misleading or generic guess.

### 3. 🔴 Off-Topic Mode (Rejected)
- **Condition**: Relevance Score < Threshold (0.60).
- **Behavior**: The system rejects the query immediately without calling the LLM generation step.
- **Example**: "How to bake a cake?" or "Who is the CEO of Apple?" (unless in docs).

## The Relevance Score (Sigmoid)

In scientific domains, raw cosine similarity scores often cluster tightly (e.g., 0.75 vs 0.82), making it difficult to distinguish between "truly relevant" and "somewhat related."

We apply a **Sigmoid Transformation** to spread these scores out. This amplifies small differences in similarity, pushing ambiguous scores toward the extremes (0 or 1). This creates a sharper decision boundary, allowing for a clean and decisive cut-off.

$$ Score = \frac{1}{1 + e^{-k(x - x_0)}} $$

- **$x$**: Average similarity of top-k documents.
- **$x_0$ (Midpoint)**: 0.50. The center of the decision boundary.
- **$k$ (Steepness)**: 18. Controls how aggressively we separate "relevant" from "irrelevant". A high steepness means a small drop in similarity results in a massive drop in score.
