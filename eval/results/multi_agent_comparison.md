# Agent configuration comparison

- agent (ma_luna_seq): `agent_ma_luna_seq_20261003_194057.json`
- agent (ma_luna_par): `agent_ma_luna_par_20261003_195744.json`
- agent (ma_sol_par): `agent_ma_sol_par_20261003_201451.json`
- agent (ma_routing): `agent_ma_routing_20261003_202907.json`
- agent (ma_workers): `agent_ma_workers_20261003_204430.json`
- agent (ma_reviewer): `agent_ma_reviewer_20261003_210817.json`

| Metric | agent (ma_luna_seq) | agent (ma_luna_par) | agent (ma_sol_par) | agent (ma_routing) | agent (ma_workers) | agent (ma_reviewer) |
|---|---|---|---|---|---|---|
| Mode accuracy | 0.98 | 0.96 | 0.98 | 0.98 | 0.98 | 0.96 |
| False rejection rate (expected RAG) | 0.0 | 0.0238 | 0.0 | 0.0 | 0.0 | 0.0238 |
| False acceptance rate (expected rejection) | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 |
| Errors | 0 | 0 | 0 | 0 | 0 | 0 |
| Unsupported claim rate | 0.0469 | 0.0407 | 0.0259 | 0.0379 | 0.0316 | 0.0233 |
| Partially supported claim rate | 0.0578 | 0.0556 | 0.0403 | 0.031 | 0.0456 | 0.0333 |
| Answers with >=1 unsupported claim | 11 | 7 | 5 | 9 | 8 | 6 |
| Answered (RAG) | 43 | 42 | 43 | 43 | 43 | 42 |
| Key-point coverage | 0.6151 | 0.6463 | 0.7103 | 0.6429 | 0.6468 | 0.6667 |
| Expected-source hit rate | 0.8333 | 0.8537 | 0.8571 | 0.8571 | 0.881 | 0.878 |
| Avg searches | 1.5 | 1.72 | 1.86 | 1.84 | 2.02 | 1.86 |
| Avg LLM calls | 3.72 | 3.62 | 3.26 | 3.48 | 5 | 4.64 |
| Latency mean (s) | 18.3652 | 19.7254 | 20.1312 | 16.7904 | 18.1914 | 28.0706 |
| Latency p50 (s) | 15.18 | 15.8 | 20.34 | 16.66 | 13.66 | 24.88 |
| Latency p90 (s) | 33.38 | 37.13 | 32.08 | 30.31 | 31.25 | 45.39 |
| Avg prompt tokens | 9411 | 10285.24 | 8245.78 | 8941.22 | 12201.54 | 12213.66 |
| Avg completion tokens | 403.68 | 438.18 | 417.22 | 417.24 | 568 | 936.68 |
| Avg cost per question (USD) | 0.0009 | 0.001 | 0.0198 | 0.0107 | 0.0057 | 0.0193 |
| Total cost (USD) | 0.0453 | 0.0496 | 0.9901 | 0.536 | 0.2854 | 0.9656 |
| Runs with an unpriced model | 0 | 0 | 0 | 0 | 0 | 0 |
| Variant-only rejections (pairs) | 0 | 0 | 0 | 0 | 0 | 0 |
| Stop reasons | {'answered': 43, 'insufficient_evidence': 3, 'out_of_domain': 4} | {'answered': 42, 'insufficient_evidence': 4, 'out_of_domain': 4} | {'answered': 43, 'insufficient_evidence': 3, 'out_of_domain': 4} | {'answered': 43, 'insufficient_evidence': 3, 'out_of_domain': 4} | {'answered': 43, 'insufficient_evidence': 3, 'out_of_domain': 4} | {'answered': 42, 'insufficient_evidence': 4, 'out_of_domain': 4} |
| Routes | {'single': 46, 'n/a': 4} | {'single': 46, 'n/a': 4} | {'single': 46, 'n/a': 4} | {'light': 27, 'heavy': 19, 'n/a': 4} | {'light': 27, 'team': 19, 'n/a': 4} | {'light': 29, 'heavy': 17, 'n/a': 4} |
| Escalations (light -> heavy) | 0 | 0 | 0 | 0 | 1 | 0 |
| Runs with parallel tool calls | 0 | 19 | 26 | 19 | 0 | 17 |
| Avg review rounds | 0 | 0 | 0 | 0 | 0 | 0.98 |

Per category: mode accuracy / false rejection / unsupported claim rate / mean latency

| Category | agent (ma_luna_seq) | agent (ma_luna_par) | agent (ma_sol_par) | agent (ma_routing) | agent (ma_workers) | agent (ma_reviewer) |
|---|---|---|---|---|---|---|
| abbreviation | 1.0 / 0.0 / 0.0 / 18.1075 | 1.0 / 0.0 / 0.0 / 20.92 | 1.0 / 0.0 / 0.0 / 20.0375 | 1.0 / 0.0 / 0.0 / 13.65 | 1.0 / 0.0 / 0.0 / 16.0325 | 1.0 / 0.0 / 0.0 / 25.3825 |
| ambiguous | 1.0 / 0.0 / 0.0 / 25.5633 | 1.0 / 0.0 / 0.0 / 31.2267 | 1.0 / 0.0 / 0.0 / 22.8667 | 1.0 / 0.0 / 0.0385 / 20.41 | 1.0 / 0.0 / 0.0476 / 21.0 | 1.0 / 0.0 / 0.0357 / 33.5233 |
| easy | 1.0 / 0.0 / 0.0476 / 15.9167 | 1.0 / 0.0 / 0.0417 / 13.125 | 1.0 / 0.0 / 0.0811 / 21.1783 | 1.0 / 0.0 / 0.1154 / 14.8567 | 1.0 / 0.0 / 0.0645 / 13.7 | 1.0 / 0.0 / 0.0333 / 25.9283 |
| multi_hop | 1.0 / 0.0 / 0.058 / 25.6644 | 1.0 / 0.0 / 0.0571 / 27.0611 | 1.0 / 0.0 / 0.0122 / 27.18 | 1.0 / 0.0 / 0.0244 / 26.9689 | 1.0 / 0.0 / 0.013 / 29.8644 | 1.0 / 0.0 / 0.0241 / 42.4178 |
| no_answer | 0.75 / None / 0.125 / 15.785 | 0.75 / None / 0.4 / 14.9 | 0.75 / None / 0.5714 / 14.14 | 0.75 / None / 0.5 / 10.94 | 0.75 / None / 0.2857 / 13.24 | 0.75 / None / 0.5 / 12.74 |
| normal | 1.0 / 0.0 / 0.0179 / 16.7611 | 1.0 / 0.0 / 0.04 / 20.1111 | 1.0 / 0.0 / 0.0 / 20.3344 | 1.0 / 0.0 / 0.0179 / 16.4911 | 1.0 / 0.0 / 0.0172 / 16.8733 | 1.0 / 0.0 / 0.0161 / 28.6267 |
| off_topic | 1.0 / None / None / 1.465 | 1.0 / None / None / 1.44 | 1.0 / None / None / 2.17 | 1.0 / None / None / 1.235 | 1.0 / None / None / 1.245 | 1.0 / None / None / 1.245 |
| paraphrase | 1.0 / 0.0 / 0.1053 / 19.59 | 1.0 / 0.0 / 0.0 / 22.5967 | 1.0 / 0.0 / 0.0 / 21.4133 | 1.0 / 0.0 / 0.0 / 17.91 | 1.0 / 0.0 / 0.0 / 21.76 | 1.0 / 0.0 / 0.0 / 26.01 |
| sequential | 1.0 / 0.0 / 0.0833 / 23.1975 | 0.75 / 0.25 / 0.1333 / 19.6025 | 1.0 / 0.0 / 0.0 / 21.67 | 1.0 / 0.0 / 0.0417 / 16.755 | 1.0 / 0.0 / 0.05 / 22.5575 | 0.75 / 0.25 / 0.0 / 41.325 |
| typo | 1.0 / 0.0 / 0.069 / 17.8125 | 1.0 / 0.0 / 0.0 / 23.5125 | 1.0 / 0.0 / 0.0323 / 21.7375 | 1.0 / 0.0 / 0.0833 / 18.49 | 1.0 / 0.0 / 0.0435 / 16.5375 | 1.0 / 0.0 / 0.0 / 26.7975 |

Questions whose mode differs:

| id | expected | agent (ma_luna_seq) | agent (ma_luna_par) | agent (ma_sol_par) | agent (ma_routing) | agent (ma_workers) | agent (ma_reviewer) |
|---|---|---|---|---|---|---|---|
| sq2 | RAG | RAG | NO_ANSWER_IN_DOCS | RAG | RAG | RAG | NO_ANSWER_IN_DOCS |
