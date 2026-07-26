# Hyperparameter Tuning

## Strategy
We optimized the RAG pipeline using a grid-search approach, evaluating performance on a set of 6 representative technical queries (ranging from core OLED physics to off-topic)

## Key Parameters Tuned
All parameters were finalized and validated in `OLED_Assistant_v6_GCP.ipynb`, where we confirmed strong relevance separation and stable Strict RAG behavior.
- **Embedding**: `BAAI/bge-m3`
- **Chunking**: `CHUNK_SIZE = 3000`, `CHUNK_OVERLAP = 500`
- **Retrieval**: `CANDIDATE_TOP_K = 20`, `MIN_DOC_RELEVANCE = 0.50`, `FINAL_TOP_N = 4`
- **Reranker**: `BAAI/bge-reranker-base`
- **Off-topic gate**: `OFF_TOPIC_THRESHOLD = 0.25`, `SIGMOID_MIDPOINT = 0.68`, `SIGMOID_STEEPNESS = 10`

> All thresholds are expressed as **relevance** = `sigmoid(cosine similarity)`.
> Raw similarity is never compared against a threshold.

### 1. Chunking Strategy
We experimented with various chunk sizes to balance context window vs. retrieval precision
- **Tested**: 800, 2000, 3000, 4500, 6000, 8000 characters
- **Winner**: **3000 chars / 500 overlap**
- **Reasoning**: OLED technical papers are dense. 800 was too short to capture full experimental setups. 8000 confused the retrieval model with multiple topics. 3000 captured roughly one distinct section/subsection

### 2. Sigmoid Function
We tuned the steepness of the relevance curve to reduce false positives
- **Parameter**: `SIGMOID_STEEPNESS`
- **Result**: Set to **10**
- **Effect**: This steepness provided better spreading, creating a sharper boundary between relevant and irrelevant queries

### 3. Threshold Measurement

Thresholds were chosen from a measured relevance distribution rather than by
intuition. `scripts/measure_relevance_distribution.py` runs representative
queries against the live ChromaDB (7,231 chunks) and prints the relevance of
every retrieved candidate. Re-run it whenever the corpus changes, because both
thresholds are corpus-dependent.

**Separation between tiers** (relevance of the single strongest document):

| Query group | Best-document relevance |
| :--- | :--- |
| On-topic, broad ("What is OLED operation principle?") | 0.766 – 0.827 |
| On-topic, narrow ("How does TADF work?") | 0.691 – 0.868 |
| On-topic, uncovered by docs ("Most OLED patents last year?") | 0.635 – 0.689 |
| **Off-topic** ("How do I bake a chocolate cake?") | **0.009 – 0.044** |

The nearly 15× gap between the weakest on-topic query (0.635) and the strongest
off-topic query (0.044) is what lets the gate stay loose.

### 4. Off-Topic Gate
- **Parameter**: `OFF_TOPIC_THRESHOLD`
- **Value**: Set to **0.25**, compared against the best document's relevance
- **Reasoning**: The maximum-margin point sits near 0.34, but the gate only
  selects a rejection message and never controls which documents reach the LLM,
  so a looser value costs nothing and protects unusually phrased questions.
  0.25 still clears the strongest off-topic query by 5.7×.
- **Superseded**: An earlier design gated on the sigmoid of the **average** of
  the top-4 raw similarities at `0.60`. Measurement showed legitimate queries
  landing at 0.603, i.e. within 0.003 of rejection, because a few weak chunks
  dragged the average down. Scoring documents individually removed that
  fragility.

### 5. Document Filter
- **Parameter**: `MIN_DOC_RELEVANCE`
- **Value**: Set to **0.50**
- **Reasoning**: This is the real precision knob. It has to prune weak chunks
  while still leaving the cross-encoder more than `FINAL_TOP_N` documents to
  choose from, otherwise reranking is skipped entirely.

| Filter value | Survivors, weakest on-topic query | Reranker runs? |
| :--- | :--- | :--- |
| 0.50 | 7 of 20 | Yes, on every tested query |
| 0.60 | 2 of 20 | No, pool collapses below Final Top-N |

  Off-topic queries retain **zero** documents even at 0.10, so the filter also
  acts as a second line of defence behind the gate.

### 6. Retrieval and Reranking
- **Candidate pool**: `CANDIDATE_TOP_K = 20`
- **Final context size**: `FINAL_TOP_N = 4`
- **Reranker**: `BAAI/bge-reranker-base`
- **Reasoning**: Fixed top-4 retrieval can miss useful supporting chunks. The updated pipeline first retrieves a wider candidate pool, keeps all sufficiently relevant documents, and then uses a BGE reranker to choose the strongest final context.

## Experiment Data
Raw experiment logs and CSV results are available in the `experiments/hyperparameters` directory.
