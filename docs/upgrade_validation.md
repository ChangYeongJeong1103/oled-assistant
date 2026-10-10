# LangGraph upgrade validation — updated 2026-10-09

## Status

The LangGraph and session-memory upgrade is merged. Implementation checks, offline regression tests, three 50-question evaluations and the 21-turn multi-turn evaluation are complete. The migration gate passed: there were no false acceptances, runtime errors or unsupported claims in the empty-history and multi-turn runs.

The historical **98% mode accuracy, 0 false acceptances, 1/289 unsupported claims** belongs to the earlier frozen combined-agent run. It is not validation of this upgrade, and a subsequent legacy refactor already separated that source snapshot from the current baseline.

## Completed checks

| Check | Result | Scope |
| --- | --- | --- |
| Drive source vs GitHub baseline | Identical inspected source/config/docs | Baseline commit `c003d9e563c9b4066deafcafeb9291888cdab0b6` |
| Dependency installation / `pip check` | Pass | Tested Python 3.11 environment; LangGraph 1.2.12, SQLite checkpointer 3.1.1 |
| Retrieval dependency parity | Exact JSON match | Two fixture documents, four chunks, three queries, fixed embeddings and fake reranker; real persisted Chroma |
| Graph/session/UI/evaluator regression tests | **27 passed** | Latest offline run: Python 3.12.14, real LangGraph/SQLite, scripted model/judge replies; no paid calls |
| Python compilation / evaluator CLI | Pass | Source, tests and scripts import/compile; both evaluators expose the intended CLI |
| Current legacy baseline | **50/50 modes correct** | One fresh run; 0 false acceptance, 0 runtime errors |
| Stateless LangGraph comparison | **49/50 modes correct** | 0 false acceptance, 0 runtime errors |
| Graph with fresh empty history | **49/50 modes correct** | 0 false acceptance, 0 unsupported claims |
| Multi-turn quality | **21/21 modes correct** | 20/21 raw interpretation judgments, 0 false acceptance, 0 unsupported claims |
| Evaluation recovery | Pass | Saved answers/judges reused, partial journal tail recovered, real SQLite/LangGraph clarification context restored |

## Live results and judge audit

| Stage | Runs | Mode accuracy | False acceptance | Raw unsupported claims | Mean / p90 latency | Cost / question |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current legacy | 50 | 100% | 0/8 | 1/295 (0.34%) | 45.9 / 76.9 s | $0.0157 |
| Stateless graph | 50 | 98% | 0/8 | 4/302 (1.32%) | 23.8 / 42.8 s | $0.0169 |
| Graph, fresh empty thread | 50 | 98% | 0/8 | 0/274 | 22.3 / 37.6 s | $0.0171 |
| Multi-turn | 21 | 100% | 0 | 0/96 | 19.7 / 38.0 s | $0.0108 |

Both graph suites rejected `sq2`, the same retrieval-sensitive question that failed in the historical frozen run. A targeted follow-up answered `sq2` correctly in all three repeats. The same repeat also ran `n05` and `e04` three times each; all nine answers were RAG with zero unsupported claims. This supports run-to-run variation rather than a deterministic LangGraph regression.

The raw judge results are preserved. The legacy run's single unsupported claim was a claim-extraction error: the judge combined the phosphorescent and TADF statements into a claim the answer did not make. Three of the stateless graph's four unsupported claims on `n05` were also absent from the answer. The remaining `e04` claim was a cautious but uncited statement that theoretical estimates are not guarantees. No raw score was overwritten.

The multi-turn interpretation judge marked 20/21 correct. Its only rejection was `clarify/3`: the planner resolved “Phosphorescence” to “Compare TADF with phosphorescence,” while the reference additionally required an explicit focus on triplet harvesting. The resolved question faithfully completed the user's pending comparison and all 21 answer modes were correct. Of the 96 judged claims, 93 were supported, 3 were partially supported and 0 were unsupported. We retain the raw 95.2% interpretation score and record this reference mismatch rather than changing the measured result.

Artifacts:

- [Legacy results](../eval/results/agent_upgrade_legacy_20261009_182249.json)
- [Stateless graph results](../eval/results/agent_upgrade_graph_20261009_183727.json)
- [Empty-history graph results](../eval/results/agent_upgrade_empty_history_20261009_185612.json)
- [Multi-turn results](../eval/results/graph_multiturn_20261009_190626.json)
- [Targeted three-repeat results](../eval/results/agent_upgrade_targeted_repeat3_20261009_191149.json)
- [Judge audit](../eval/results/upgrade_judge_audit_20261010.json)

These are single runs on development data, not a held-out benchmark or proof of production quality. Questions ran sequentially with four PyTorch threads, a shared retriever, unchanged models/retrieval settings, the 150-second deadline and judge policy `scoped_absence_v2`. An initial six-answer setup pilot and a ten-answer concurrent load pilot are excluded; the load pilot produced a resource-sensitive deadline failure. The targeted three-repeat check is diagnostic and not a replacement for repeated full-suite evaluation.

The targeted-repeat manifest has a different corpus SQLite file hash (`8ee87689...`) from the main suites (`f4c1ab49...`). A binary SQLite hash can change without a change to document text, but logical corpus equivalence has not been verified. The repeat demonstrates successful answers in that run; it does not by itself establish a controlled same-corpus comparison.

After the measured runs, failure diagnostics were strengthened without changing answer behavior: an existing ChromaDB is no longer deleted after a generic open failure, and private results now keep the complete retrieved ledger even when the final mode is `NO_ANSWER_IN_DOCS`. Future evaluations use `separate_grounding_coverage_v3`, which prevents expected points from entering claim extraction and requires every extracted claim to quote the answer. The raw v2 measurements above remain unchanged.

The v3 evaluator is checked offline, not yet measured with a live judge in this patch. An empty extraction or an invalid/missing v3 answer quote makes `grounding_valid=false` for the answer. Summaries expose `grounding_invalid_answers` and `grounding_evaluation_complete`; a low unsupported rate is not a passed grounding gate when extraction is incomplete. A valid automated grounding evaluation requires `grounding_evaluation_complete=true` and `judge_extraction_errors=0`. These fields describe evaluation validity, not proof that all claims are correct.

`eval/conversations.json` remains the frozen v1 rubric. `eval/conversations_v2.json` is the default for new multi-turn runs: it preserves all user turns and mode labels, and corrects only the `clarify/3` reference intent and interpretation criterion. The earlier 20/21 interpretation result belongs to v1 and is not a measured v2 result.

The dependency fixture compares chunk text/metadata, candidate order and relevance rounded to seven decimals, filter survivors, and reranked output. Old dependencies: LangChain 0.2.17, community 0.2.19, core 0.2.43, text-splitters 0.2.4. New dependencies: community 0.4.2, core 1.6.9, text-splitters 1.1.3. Chroma 1.5.9 was held constant. Both result files have SHA-256 `9a122375d6dcf07527c570f2b313f428ceee4a3fa8c3fb451d447277f5684bc0`. This proves compatibility on the fixture, not full-corpus BGE ranking parity. `scripts/check_retrieval_compat.py` makes the check reproducible.

The 27 tests cover:

- Exact legacy/graph model-call payload sequence for simple/heavy routes, off-topic exit, citation correction, review revision and light-to-heavy escalation.
- Team revision returns to the same writer; the two workers are not restarted.
- Deadline and LLM budget accounting across graph node boundaries.
- SQLite close/reopen, distinct-thread isolation, fresh request counters, and a five-pair history window.
- Clarification retains unresolved intent; invalid interpretation fails closed; old history cannot supply valid evidence for a new answer.
- Streamlit streaming, interpreted-question display, clarification display and New chat resetting the thread/history.
- A ChromaDB open failure preserves the existing directory, and a refusal retains retrieved evidence for private diagnostics.
- Public evaluation export strips text from cited and retrieved evidence; invalid judge claim extraction is reported separately from grounding scores.
- Grounding and coverage receive separate inputs; missing v3 quotes and empty extractions cannot pass the evaluation-validity gate.
- Saved single-turn and multi-turn rejudging preserves original files, agent answers/latency/cost and execution metadata while refreshing judge metrics and costs.
- Conversation v2 preserves user turns and mode labels; historical rubric selection uses the saved hash, and missing raw evidence is rejected before API calls.

## Reproduce offline checks

```bash
pip install -r requirements-dev.txt
python -m pip check
python -m pytest -q tests
python -m compileall -q src scripts tests
```

For dependency parity, run the fixture in separate old/new dependency environments against the same temporary Chroma directory and compare the JSON. The fixture disables neural model loading; it must be run as its own process.

## Re-run the live evaluation

`scripts/evaluate_upgrade.py` loads the retriever once and runs the three 50-question suites followed by the 21-turn development set. It reuses completed answers and exact-answer judge results from JSONL journals, preserving their original latency/usage. Multi-turn recovery restores bounded history and pending clarification into an isolated thread; it does not reuse prior retrieval evidence. Source/data hashes and effective non-secret runtime settings must match before cached records are reused. Interrupted calls without a saved result may need to run again.

The saved recovery files are under `eval/results/raw/upgrade_journal/` in Drive and belong to the exact evaluated source hashes. Because post-evaluation source files have changed, use a new journal directory for another trial:

```bash
python scripts/evaluate_upgrade.py --run --journal-dir eval/results/raw/upgrade_trial_2
```

Do not mix pilot journals into this recovery directory.

## Individual evaluation commands

These commands call OpenAI with questions, conversation context, retrieved passages and generated answers. Use the project's configured key and corpus. Keep models, prompts, retrieval settings and judge policy identical across engine comparisons. Start with one run; repeat before attributing small differences to the graph migration.

```bash
# Current legacy baseline, after dependency migration
python scripts/evaluate_agent.py --run --engine legacy --label legacy_baseline

# Stage 1: graph, no memory
python scripts/evaluate_agent.py --run --engine graph --label graph_stateless

# Stage 2: checkpointer enabled, fresh empty thread for every question
python scripts/evaluate_agent.py --run --engine graph \
  --session-db sessions/eval.sqlite --label graph_empty_history

# Contextual interpretation and grounded answers, measured separately
python scripts/evaluate_multiturn.py --repeat 3

# Explicit historical v1 rubric
python scripts/evaluate_multiturn.py --dataset eval/conversations.json --repeat 1

# Replace these paths with generated results
python scripts/evaluate_agent.py --compare \
  eval/results/LEGACY.json eval/results/GRAPH.json --out langgraph_comparison.md
```

Both conversation rubric versions contain **9 development conversations / 21 turns**: pronouns, omitted subjects, explicit topic switches, OFF_TOPIC transitions, follow-ups after NO_ANSWER, clarification and resolution/cancellation, unnecessary-clarification checks, and Korean follow-ups. Reference intents are visible only to the evaluator. Neither conversation set nor the 50-question set is a held-out benchmark.

Interpretation accuracy and clarification rates are separate from final mode accuracy and claim-grounding rates. A correct interpretation followed by a wrong answer mode is counted separately. Result manifests record code/data hashes, dependency versions and judge configuration. Raw evidence stays under git-ignored `eval/results/raw`; public result copies omit chunk text. Agent, grounding-judge and interpretation-judge costs are reported separately.

## Rejudge saved answers without rerunning the agent

`--rejudge` supports single-turn and multi-turn results. For a public result, its matching raw file must be available. Multi-turn rejudging resolves conversation/turn IDs and refreshes both grounding and interpretation metrics, including the two judge costs. The default conversation rubric must match the recorded dataset hash; use `--dataset` to explicitly change the rubric.

```bash
# Same saved answers, current grounding judge, original conversation rubric
python scripts/evaluate_agent.py --rejudge eval/results/graph_multiturn_20261009_190626.json

# Same answers, explicitly grade against the corrected v2 interpretation rubric
python scripts/evaluate_agent.py --rejudge eval/results/graph_multiturn_20261009_190626.json \
  --dataset eval/conversations_v2.json
```

These commands still make paid judge calls; they were not executed for this patch. Original result files are preserved and rejudged results get new names. Rejudged files use `engine_manifest` for the original execution and `judge_manifest` for the new policy/model, evaluator source hashes and rubric hashes. The ambiguous top-level `manifest` is removed only from the new rejudged copy. Saved answers, agent usage and latency are retained.

## Remaining boundaries

- Clarification initially applies to contextual follow-ups. Empty-history requests retain the original planner prompt/schema for regression parity.
- SQLite supports the current single-process deployment. Durable container sessions require a persistent volume; the existing Cloud Run/Kubernetes configuration does not provide one. Shared multi-instance sessions need a server database such as Postgres and session authorization.
- Checkpoints preserve backend conversations, but the UI does not yet list or reopen old chats after a browser session ends. New chat starts a new thread.
- Research/review remain one coarse node. Mid-request crash replay, `Send` workers and finer-grained review edges are deferred.
- Existing experimental notebooks target their historical dependencies; the current requirements target the application and evaluation scripts.
