# LLM Comparison: Mistral-Nemo vs GPT-4o-mini

> **Note**: This is an early-stage record from the on-premise phase of the project, kept for reference. It does not describe the current system, which uses a GPT-6-Luna / GPT-6.1-Sol agent.

## Decision at the Time: GPT-4o-mini (Cloud Deployment) & Mistral-Nemo (Local)

At this stage, we selected **GPT-4o-mini** as the default model for the cloud deployment track.
For the internal/private deployment track, we kept **Mistral-Nemo** as a local serving option.

### Why GPT-4o-mini at the Time?
1.  **Cloud deployment simplicity**: Works directly with managed platforms (e.g., GCP) without hosting a separate model server.
2.  **Low latency in deployment**: Supports fast response times in production-style cloud runs.
3.  **Operational consistency**: Aligns with the deployment pattern already proven in `trading-advisor`.

### Why Mistral?
1.  **Security**: No data leaves the local computer. This is critical for proprietary internal data.
2.  **Cost**: $0 operational cost (runs on local computer).
3.  **Performance**: For RAG tasks (which rely heavily on the *context* provided, not the model's world knowledge), Mistral-Nemo demonstrated performance comparable to GPT-4o-mini in summarization quality.

### Comparison Results

We conducted a side-by-side evaluation using an LLM-as-a-judge approach (scoring Specificity, Relevance, and Factuality)

Results below are taken from `experiments/llm_comparison/hyperparameter_experiments_Mistral_vs_GPT.csv`
(historical comparison configuration: `CHUNK_SIZE=3000`, `CHUNK_OVERLAP=500`, `TOP_K=4`, `RELEVANCE_THRESHOLD=0.6`, `SIGMOID=(0.5, 18)`).

| Metric (RAG answers only) | GPT-4o-mini (Cloud) | Mistral-Nemo (Local) |
| :--- | :---: | :---: |
| **Specificity** | 9.00 / 10 | 8.00 / 10 |
| **Relevance** | 10.00 / 10 | 8.33 / 10 |
| **Factuality** | 9.00 / 10 | 6.67 / 10 |
| **Avg Latency (sec)** | 2.98 | 17.42 |

Notes:
- Mistral answered **3/6** test queries in RAG mode, slightly more than GPT (2/6). This does not mean Mistral is natively superior, but rather that our **Prompt Optimization** and **Hyperparameter Tuning** were highly effective for the local model.
- GPT-4o-mini could reach similar or better quality with dedicated tuning while providing better cloud deployment characteristics.
- We intentionally kept the local deployment goal: validating that an internal, privacy-first Mistral stack could still deliver production-grade Strict RAG quality.

**Conclusion at the time:** Both models were viable for Strict RAG. GPT-4o-mini became the cloud default because of its deployment speed and lower latency, and Mistral-Nemo remained the privacy-first local option. Both choices were later superseded, as described in the note at the top of this page.

## Detailed Logs
Detailed comparison notebooks and logs can be found in `experiments/llm_comparison` directory.
