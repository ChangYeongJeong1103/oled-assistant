# Hyperparameter Tuning

## Strategy

We optimized the RAG pipeline using a grid-search approach, evaluating performance on a set of 6 representative technical queries (ranging from core OLED physics to off-topic)

## Key Parameters Tuned

All parameters were finalized and validated in `OLED_assistant_v6_GCP.ipynb`, where we confirmed strong relevance separation and stable Strict RAG behavior.

- **Embedding**: `BAAI/bge-m3`
- **Chunking**: `CHUNK_SIZE = 3000`, `CHUNK_OVERLAP = 500`
- **Retrieval**: `CANDIDATE_TOP_K = 20`, `MIN_DOC_RELEVANCE = 0.50`, `FINAL_TOP_N = 4`
- **Reranker**: `BAAI/bge-reranker-base`
- **Relevance scale**: `SIGMOID_MIDPOINT = 0.60`, `SIGMOID_STEEPNESS = 17.27`

> All thresholds are expressed as **relevance** = `sigmoid(cosine similarity)`. Raw similarity is never compared against a threshold.

### 1. Chunking Strategy

We experimented with various chunk sizes to balance context window vs. retrieval precision

- **Tested**: 800, 2000, 3000, 4500, 6000, 8000 characters
- **Winner**: **3000 chars / 500 overlap**
- **Reasoning**: OLED technical papers are dense. 800 was too short to capture full experimental setups. 8000 confused the retrieval model with multiple topics. 3000 captured roughly one distinct section/subsection

### 2. Sigmoid Function

We tuned the steepness of the relevance curve to reduce false positives

- **Parameter**: `SIGMOID_STEEPNESS`
- **Result**: Originally tuned to **10** (midpoint 0.68)
- **Effect**: This steepness provided better spreading, creating a sharper boundary between relevant and irrelevant queries
- **Recalibrated**: The distance-to-cosine formula was corrected to `cos = 1 - d/2` (Chroma returns squared L2 distance). The sigmoid was refit to midpoint **0.60** and steepness **17.27** so that relevance 0.50 and 0.25 fall on the same documents as before; no decision changed. See [Retrieval](retrieval.md#distance--cosine-corrected).

### 3. Threshold Measurement

Thresholds were chosen from a measured relevance distribution rather than by intuition. `scripts/measure_relevance_distribution.py` runs representative queries against the live ChromaDB (7,231 chunks) and prints the relevance of every retrieved candidate. Re-run it whenever the corpus changes, because the threshold is corpus-dependent.

**Separation between query groups** (relevance of the single strongest document):

| Query group | Best-document relevance |
| :-- | :-- |
| On-topic, broad ("What is OLED operation principle?") | 0.806 – 0.877 |
| On-topic, narrow ("How does TADF work?") | 0.717 – 0.922 |
| On-topic, uncovered by docs ("Most OLED patents last year?") | 0.652 – 0.716 |
| **Off-topic** ("How do I bake a chocolate cake?") | **0.019 – 0.059** |

There is an 11× gap between the weakest on-topic query (0.652) and the strongest off-topic query (0.059). (Values measured after the distance-formula correction; the decisions at 0.50 are unchanged.) Off-topic questions are classified by the agent's planner before any search, so this gap is not used as a gate. It shows that the filter would return nothing for them anyway.

Scoring documents individually also matters here. An earlier design scored the **average** of the top-4 similarities, and legitimate narrow queries landed within 0.003 of rejection because a few weak chunks dragged the average down.

### 4. Document Filter

- **Parameter**: `MIN_DOC_RELEVANCE`
- **Value**: Set to **0.50**
- **Reasoning**: This is the real precision knob. It has to prune weak chunks while still leaving the cross-encoder more than `FINAL_TOP_N` documents to choose from, otherwise reranking is skipped entirely.

| Filter value | Survivors, weakest on-topic query | Reranker runs? |
| :-- | :-- | :-- |
| 0.50 | 7 of 20 | Yes, on every tested query |
| 0.60 | 2 of 20 | No, pool collapses below Final Top-N |

Off-topic queries retain **zero** documents even at 0.10, so the filter also acts as a second line of defence behind the planner.

### 5. Retrieval and Reranking

- **Candidate pool**: `CANDIDATE_TOP_K = 20`
- **Final context size**: `FINAL_TOP_N = 4`
- **Reranker**: `BAAI/bge-reranker-base`
- **Reasoning**: Fixed top-4 retrieval can miss useful supporting chunks. The updated pipeline first retrieves a wider candidate pool, keeps all sufficiently relevant documents, and then uses a BGE reranker to choose the strongest final context.

## Experiment Data

Raw experiment logs and CSV results are available in the `experiments/hyperparameters` directory.
