# Engine comparison

Left: `workflow_baseline_20261003_165849.json`  
Right: `agent_v3_20261003_182356.json`

| Metric | workflow (baseline) | agent (v3) |
|---|---|---|
| Mode accuracy | 0.8824 | 0.9706 |
| False rejection rate (expected RAG) | 0.1538 | 0.0 |
| False acceptance rate (expected rejection) | 0.0 | 0.125 |
| Errors | 0 | 0 |
| Unsupported claim rate | 0.1344 | 0.0508 |
| Partially supported claim rate | 0.2043 | 0.1211 |
| Answers with >=1 unsupported claim | 8 | 10 |
| Answered (RAG) | 22 | 27 |
| Key-point coverage | 0.6439 | 0.7692 |
| Expected-source hit rate | 0.7727 | 0.8462 |
| Avg searches | 1 | 1.1176 |
| Avg LLM calls | 0.7353 | 3.0294 |
| Latency mean (s) | 6.1659 | 10.9926 |
| Latency p50 (s) | 8.15 | 11.08 |
| Latency p90 (s) | 9.88 | 18.65 |
| Avg prompt tokens | 2197.0588 | 5228.7941 |
| Avg completion tokens | 145.2647 | 537.6765 |
| Avg cost per question (USD) | 0.0008 | 0.0023 |
| Total cost (USD) | 0.0286 | 0.0774 |
| Variant-only rejections (pairs) | 3 | 0 |
| Stop reasons | {'rag': 22, 'no_answer_in_docs': 7, 'off_topic': 5} | {'answered': 27, 'insufficient_evidence': 3, 'out_of_domain': 4} |

| Category | workflow (baseline) acc / false rej / unsup | agent (v3) acc / false rej / unsup |
|---|---|---|
| abbreviation | 1.0 / 0.0 / 0.2791 | 1.0 / 0.0 / 0.0 |
| ambiguous | 0.0 / 1.0 / None | 1.0 / 0.0 / 0.0312 |
| multi_hop | 1.0 / 0.0 / 0.1304 | 1.0 / 0.0 / 0.0667 |
| no_answer | 1.0 / None / None | 0.75 / None / 0.0 |
| normal | 1.0 / 0.0 / 0.0141 | 1.0 / 0.0 / 0.05 |
| off_topic | 1.0 / None / None | 1.0 / None / None |
| paraphrase | 1.0 / 0.0 / 0.1667 | 1.0 / 0.0 / 0.0714 |
| typo | 0.75 / 0.25 / 0.2 | 1.0 / 0.0 / 0.1212 |

| id | expected | workflow (baseline) | agent (v3) |
|---|---|---|---|
| t01 | RAG | NO_ANSWER_IN_DOCS | RAG |
| amb1 | RAG | NO_ANSWER_IN_DOCS | RAG |
| amb2 | RAG | NO_ANSWER_IN_DOCS | RAG |
| amb3 | RAG | OFF_TOPIC | RAG |
| na1 | NO_ANSWER_IN_DOCS | NO_ANSWER_IN_DOCS | RAG |
