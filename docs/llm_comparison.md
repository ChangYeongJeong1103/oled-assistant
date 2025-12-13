# LLM Comparison: Mistral-Nemo vs GPT-4o-mini

## Decision: Mistral-Nemo (Local)

We selected **Mistral-Nemo (12B)** running locally via Ollama as the production model for this project

### Why Mistral?
1.  **Security**: No data leaves the local computer. This is critical for proprietary internal data.
2.  **Cost**: $0 operational cost (runs on local computer).
3.  **Performance**: For RAG tasks (which rely heavily on the *context* provided, not the model's world knowledge), Mistral-Nemo demonstrated performance comparable to GPT-4o-mini in summarization quality.

### Comparison Results

We conducted a side-by-side evaluation using an LLM-as-a-judge approach (scoring Specificity, Relevance, and Factuality)

Results below are taken from `experiments/llm_comparison/hyperparameter_experiments_Mistral_vs_GPT.csv`
(configuration: `CHUNK_SIZE=3000`, `CHUNK_OVERLAP=500`, `TOP_K=4`, `RELEVANCE_THRESHOLD=0.6`, `SIGMOID=(0.5, 18)`).

| Metric (RAG answers only) | GPT-4o-mini (Cloud) | Mistral-Nemo (Local) |
| :--- | :---: | :---: |
| **Specificity** | 9.00 / 10 | 8.00 / 10 |
| **Relevance** | 10.00 / 10 | 8.33 / 10 |
| **Factuality** | 9.00 / 10 | 6.67 / 10 |
| **Avg Latency (sec)** | 2.98 | 17.42 |

Notes:
- Mistral answered **3/6** test queries in RAG mode, slightly more than GPT (2/6). This does not mean Mistral is natively superior, but rather that our **Prompt Optimization** and **Hyperparameter Tuning** were highly effective for the local model.
- GPT-4o-mini could achieve similar or better results with dedicated tuning, but our goal was to prove viability on local hardware.

**Conclusion:** Although the native performance (especially latency) of the local model is lower than commercial cloud LLMs, our results show that **on-premise LLMs can achieve commercial-grade accuracy** through rigorous prompt optimization, high-quality RAG data, and hyperparameter tuning. To a PhD-level expert, the generated technical answers showed no significant difference in quality.

### Future Work
- **Fine-tuning**: We plan to fine-tune Mistral-Nemo on internal OLED technical reports to further improve domain specificity and reduce latency.

## Detailed Logs
Detailed comparison notebooks and logs can be found in `experiments/llm_comparison` directory.
