# Hyperparameter Tuning

## Strategy
We optimized the RAG pipeline using a grid-search approach, evaluating performance on a set of 6 representative technical queries (ranging from core OLED physics to off-topic)

## Key Parameters Tuned
All parameters were finalized and validated in `OLED_Assistant_v6_GCP.ipynb`, where we confirmed strong relevance separation and stable Strict RAG behavior.
- **Embedding**: `BAAI/bge-m3`
- **Chunking**: `CHUNK_SIZE = 3000`, `CHUNK_OVERLAP = 500`
- **Retrieval**: `TOP_K_DOCUMENTS = 4`
- **Relevance gate**: `RELEVANCE_THRESHOLD = 0.6`, `SIGMOID_MIDPOINT = 0.68`, `SIGMOID_STEEPNESS = 10`

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

### 3. Relevance Threshold
- **Parameter**: `RELEVANCE_THRESHOLD`
- **Value**: Set to **0.60**
- **Observation**: 
  - **0.60**: "Sweet spot" for balancing precision and recall
  - **< 0.60**: Starts to accept generic or slightly irrelevant questions.
  - **> 0.60**: Starts to reject legitimate OLED questions just because they were phrased differently
  
## Experiment Data
Raw experiment logs and CSV results are available in the `experiments/hyperparameters` directory.
