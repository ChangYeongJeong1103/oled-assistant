## AI-Driven OLED Assistant – LLM Comparison (Mistral-Nemo vs GPT-4o-mini)

### 1. Shared RAG Configuration

- **Embedding Model**: `BAAI/bge-m3`
- **Chunking Settings**: `CHUNK_SIZE = 3000`, `CHUNK_OVERLAP = 500`
- **Vector DB**: ChromaDB (same DB, same document set)
- **Retrieval Settings**: `TOP_K_DOCUMENTS = 4`
- **LLM Temperature**: `LLM_TEMPERATURE = 0.2` (deterministic, fact-focused)
- **Strict RAG Logic**:
  - 3-tier decision system: **RAG / NO_ANSWER_IN_DOCS / OFF_TOPIC**
  - Relevance Score = average similarity → sigmoid transformation
  - Thresholds: `RELEVANCE_THRESHOLD = 0.6`, `SIGMOID_MIDPOINT = 0.5`, `SIGMOID_STEEPNESS = 18`
- **Test Queries (Q1–Q6)**:
  - Q1–Q2: OLED / organic semiconductor core physics (Positive, high Relevance expected)
  - Q3–Q4: Display/semiconductor but non-OLED (medium Relevance or NO_ANSWER expected)
  - Q5–Q6: Off-topic / Non-science (OFF_TOPIC auto-rejection expected)

In other words, **the RAG pipeline and hyperparameters are completely identical**, and we compared by **only swapping the LLM**.

---

### 2. Model Configurations

- **Mistral-RAG**
  - `LLM_PROVIDER = "llama_local"` (Ollama / OpenAI-compatible server)
  - `LLM_MODEL = "mistral-nemo"`
  - Inference cost: **Local execution ($0 in CSV)**

- **GPT-4o-mini-RAG**
  - `LLM_PROVIDER = "openai"`
  - `LLM_MODEL = "gpt-4o-mini"`
  - Inference cost: Estimated based on OpenAI API pricing (per CSV)

Both experiments used the same Strict RAG adapter (`StrictRAGAdvisor`) and the same evaluation pipeline (`RAGEvaluator`, `monitored_query`) from `OLED_assistant_v2_*.ipynb`.

---

### 3. Quantitative Comparison Summary

Source: `Mistral_vs_GPT/hyperparameter_experiments_Mistral_vs_GPT.csv`

#### 3-1. Summary Table

| Metric                                   | Mistral-RAG                         | GPT-4o-mini-RAG                     |
|------------------------------------------|-------------------------------------|-------------------------------------|
| Avg Overall Score (1–10)                 | **7.67**                            | **9.33**                            |
| Specificity / Relevance / Factuality     | 8.00 / 8.33 / 6.67                  | 9.00 / 10.00 / 9.00                 |
| Avg Response Time (sec)                  | 17.42                               | 2.98                                |
| Total Tokens                             | 833                                 | 632                                 |
| Estimated Cost (USD)                     | ~$0.0000 (local, $0)                | ~$0.0003                            |
| RAG / No-Answer / Off-Topic (count)      | 3 / 1 / 2                           | 2 / 2 / 2                           |
| Avg Relevance Score (0–1, all Q1–Q6)     | 0.655                               | 0.656                               |
| Relevance mean (Q1–Q2, positive)         | 0.919                               | 0.919                               |
| Relevance mean (Q5–Q6, negative)         | 0.329                               | 0.345                               |
| Relevance gap (mean_pos – mean_neg)      | 0.590                               | 0.574                               |
| Relevance margin (min_pos – max_neg)     | 0.475                               | 0.475                               |

#### 3-2. Key Insights (Conclusion Based on Numbers)

- **Answer Quality (RAG Score)**  
  - GPT-4o-mini-RAG shows **superiority in overall quality score (9.33 vs 7.67)** and all detailed metrics (Specificity/Relevance/Factuality).  
  - Particularly in **Relevance (how directly the answer addresses the question)**, it scored a perfect 10, responding more directly to the question's intent.

- **Speed (Latency)**  
  - GPT-4o-mini-RAG averaged **2.98 seconds**, while Mistral-RAG averaged **17.42 seconds**,  
    showing that GPT-4o-mini is **approximately 5–6 times faster**.

- **Tokens & Cost**  
  - Both models used only **hundreds of tokens** for 6 queries (833 vs 632).  
  - GPT-4o-mini's estimated cost is **around $0.0003**, which is essentially negligible,  
    while Mistral is recorded as **$0.0000 (assumed $0)** since it's a local model.

- **RAG Retrieval Quality (Relevance Separation)**  
  - The average Relevance Score and **Q1–2 (OLED core) vs Q5–6 (off-topic)** separation metrics (`rel_gap`, `rel_margin`) show **only very minor differences at the decimal level** between the two experiments.  
  - This indicates that **the embedding/vector search/RAG pipeline is well-tuned**, and under the same embedding/DB settings,  
    **document relevance separation is essentially at the same level regardless of LLM choice**.
  - Therefore, it is reasonable to conclude that **most of the actual quality difference comes from "how the LLM reads documents and constructs answers"**.

---

### 4. Qualitative Comparison

Reference: `Answer_comparison.md` (Q1–Q3 answer samples)

#### 4-1. Q1 – Blue Phosphorescent OLED Degradation

- **Question**:  
  *"What are the key degradation mechanisms of a blue phosphorescent OLED?"*

- **Mistral-RAG**  
  - Covers **all key mechanisms identified in papers**, including FIrpic decomposition, CO₂ loss, complex formation with TPBi, BPhen dimer/adduct, isomerization, etc.  
  - Organizes information in a **concise bullet list format** that compresses core information, focusing on points frequently mentioned in the literature.

- **GPT-4o-mini-RAG**  
  - Provides **paper-level structured explanations** for FIrpic-based devices, including picolinate ligand cleavage, reversible/irreversible pathways,  
    isomerization, HTL degradation, BPhen dimer/adduct, etc.  
  - Compared to Mistral, answers are **slightly longer and more narrative**, but organized as **structured item-by-item explanations**,  
    making them suitable for direct use in documentation/interviews.

**Summary**: Both models provide **scientifically valid answers at the paper level**,  
with style differences being that **Mistral-RAG is more compressed and bullet-focused**, while GPT-4o-mini-RAG is **slightly longer and more narrative**.

#### 4-2. Q2 – Exciton Diffusion Length in Organic Semiconductors

- **Question**:  
  *"How does exciton diffusion length affect charge separation efficiency in organic semiconductor devices?"*

- **Mistral-RAG**  
  - Accurately identifies and explains **core physical concepts**, such as the need for excitons to reach dissociation sites/heterojunctions,  
    increased recombination with shorter diffusion lengths, and improved charge separation efficiency with longer lengths.  
  - Provides **correct answers that well reflect actual physical phenomena** based on OPV vs OLED contexts.

- **GPT-4o-mini-RAG**  
  - Has a similar structure but **more clearly emphasizes the OPV context**,  
    and explains in a slightly more **teaching-oriented style** the point that excitons must move from tight binding states to HJs.

**Summary**: Both models provide **very good answers at nearly equivalent levels** here,  
with differences mainly in **narrative style (slightly more teaching vs slightly more condensed)**.

#### 4-3. Q3 – MicroLED Mass Transfer Challenges (Out-of-domain but Display-related)

- **Question**:  
  *"What are the major challenges in mass transfer processes for MicroLED displays?"*

- **Mistral-RAG**  
  - Although there is no direct content about MicroLED mass transfer in the documents,  
    it provides **reasonably identified answers** on issues that are also important in actual MicroLED processes, such as **patterning, uniformity, contamination, scalability**,  
    based on various display-related data.  
  - From a Strict RAG perspective, this is closer to a **"indirect inference" answer based on documents**,  
    but the content itself fairly accurately reflects the core challenges of MicroLED mass transfer.

- **GPT-4o-mini-RAG**  
  - Although the Relevance Score is at a medium level (0.612),  
    the final mode was processed as **No Answer (no relevant content in documents)**.  
  - In other words, it showed behavior that **more strictly adheres to the policy of "not answering what is not in my RAG documents"**.

**Summary**:  
- Mistral-RAG has a **style of expanding context to pull in general display process knowledge**,  
- GPT-4o-mini-RAG **more faithfully follows Strict RAG policy, boldly choosing NO_ANSWER when the RAG document basis is weak**.

This difference can serve as a **model selection criterion** for **how aggressively to apply Strict RAG policy**.

---

### 5. Interpretation & Recommendations

#### 5-1. LLM Quality/Speed/Cost Tradeoffs

- **Quality (LLM-as-a-judge basis)**: GPT-4o-mini-RAG recorded **overall higher RAG evaluation scores (9.33 vs 7.67)** compared to Mistral-RAG.  
  The difference is particularly large in **question relevance (Relevance) and factuality (Factuality)**.  
  However, **when actual domain experts with OLED PhD backgrounds directly compared Q1–Q3 answers, both models provided valid answers at the paper level, and from a practical application perspective, the perceived quality difference was not significant.**
- **Speed**: GPT-4o-mini-RAG has an **average response time of 2.98 seconds**,  
  providing **very comfortable interaction speed** for use in prototypes/demos/interviews.
- **Cost**: At this experiment scale (6 dev queries), both models have **costs essentially close to $0**,  
  and even in actual projects, **the cost level allows sufficient use of GPT-4o-mini during tuning/testing phases**.

#### 5-2. Deployment/Confidentiality Perspective

- **Mistral-RAG (Local)**:
  - Suitable for **complete on-premise deployment** using company internal confidential documents.  
  - Since models/vector DBs/applications all operate within internal infrastructure, **there is no concern about data leakage externally even in scenarios where company internal information is put into RAG and shared/collaborated with multiple internal teams/colleagues.**
  - Although speed is somewhat slower, **the fact that data never leaves externally** is a major advantage, and there is room to reduce latency through future GPU/server optimization.

- **GPT-4o-mini-RAG (Cloud)**:
  - A strong option when **highest quality + fast response** is needed,  
    and external API use is permitted after data anonymization/partial masking.

---

### 6. Conclusion

- **Successful RAG Pipeline Validation**  
  - Both models showed excellent Relevance separation and OLED core question (Q1–Q2) answer quality. This indicates that **the embedding and Retrieval pipeline is already stabilized**.

- **LLM Quality Evaluation: "Judge vs Expert"**  
  - **LLM-as-a-judge (quantitative evaluation)**: GPT-4o-mini recorded higher scores (9.33 vs 7.67).
  - **Expert Review (qualitative evaluation)**: When **OLED PhD domain experts** directly reviewed answers, **both models generated answers with scientific validity at the paper level**, and from a practical application perspective, **the perceived quality difference was not significant.**
  - In other words, it has been proven that **in a Strict RAG environment, local models (Mistral) can also secure professional answer quality comparable to commercial models (GPT)**.

- **Final Deployment**  
  - **1. Prototyping & Benchmarking**: **GPT-4o-mini**  
    → Suitable for establishing the system's baseline using fast response speed and excellent Judge scores, and for use in demos/interviews.
  - **2. Actual Service & On-Premise Deployment**: **Mistral (Local LLM)**  
    → Has **sufficient answer quality validated by domain experts**, and most importantly, has the decisive advantage of being able to **safely utilize company internal confidential data without external leakage**.
  - Therefore, this project pursues **Mistral-based on-premise deployment** as the final goal, adopting a strategy that maintains both security and quality by directly porting the current Strict RAG pipeline.
