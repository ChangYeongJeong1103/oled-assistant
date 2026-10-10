# LangGraph and session memory: system design

## Goal and interfaces

Preserve Strict RAG while adding contextual follow-ups. LangChain provides the document/retrieval integrations; LangGraph controls workflow and persistence.

```python
agent = GraphAgentAssistant.with_sqlite(retriever)
agent.query("How does TADF work?", thread_id=session_id)
agent.query("Why must its energy gap be small?", thread_id=session_id)
```

Omitting `thread_id` uses a separately compiled stateless graph. A supplied ID requires a checkpointer. `SESSION_MEMORY_ENABLED=false` disables memory while retaining graph orchestration. `AgentAssistant` remains the legacy evaluation baseline. `stream()` yields progress `event` items and one final `result` item. Only Streamlit's thread renders UI; worker events use the existing event queue.

## State contract

- `original_question`: exact user input.
- `history`: five bounded user/assistant pairs, including interpreted question and response mode, never prior evidence chunks or traces.
- `pending_clarification`: unresolved full question and the clarification asked.
- `run_data`: JSON-safe `AgentRun` snapshot with current evidence, counters and deadline origin; cleared at initialization/completion.
- `result`: existing answer/source/trace shape plus original/standalone questions.

Nodes restore `AgentRun` with fresh process-local locks. Ledger entries retain text/metadata but omit LangChain Document objects. Clients, retrievers, locks and database connections are never serialized. Node transitions do not renew the deadline. Public calls start new requests; interrupted-call replay is not exposed, so a saved monotonic deadline is never resumed in a new process.

## Control flow

Only the planner sees history. It interprets the question before domain/route selection. Every downstream role sees the resolved question and new evidence. The route edge selects an existing research loop or team function. Escalation uses the existing failure policy, at most once, retaining evidence/budgets and starting a fresh heavy-model conversation.

Review stays inside `answer_handler`/`research`: feedback returns to the same outstanding `submit_answer` call. No worker restart occurs during revision.

## Clarification and errors

Contextual ambiguity is different from missing evidence. Ask only when an essential subject/comparison target cannot be resolved, not merely because a question is short. Preserve the unresolved intent across a short user reply; discard it on explicit topic change. Empty-history requests retain the old planner behavior; clarification initially applies to contextual follow-ups.

API errors do not enter history or erase a pending clarification. Previous refusals identify conversational topics but do not prove corpus-wide absence. Previous answers are never accepted as evidence, even when correctly cited in an earlier turn.

## Concurrency and storage

SQLite uses WAL, a busy timeout and the checkpointer's synchronization. Same-thread requests are serialized inside the process to prevent lost history updates. Different threads have independent request state. Retrieval retains the original shared CPU lock.

Session files are excluded from git, Docker and Cloud Run upload contexts. Use a persistent volume for container replacement. No cross-user memory, long-term fact store, automatic old-chat discovery or retention job is added. Historical checkpoints can contain question/answer/evidence text; operators should apply an appropriate storage retention policy.

## Validation design

1. Compare old/new dependency environments on a fixed-embedding Chroma fixture.
2. Compare scripted legacy/graph API call sequences for validation, review, escalation, teams, deadlines and budgets using real LangGraph and SQLite.
3. Run the current legacy and graph engines on the same 50 questions and judge policy. Historical 98% is mode accuracy, not factual answer accuracy.
4. Grade multi-turn interpretation separately from grounded answer quality. Reference intents are evaluator-only and never enter agent inputs.

Both question sets are development/regression data, not held-out benchmarks. Repeat stochastic runs before attributing small metric differences to a change.

Validation is complete: the current legacy baseline and both graph configurations ran all 50 questions, followed by the 21-turn multi-turn set. The graph had no false acceptances or runtime errors; the empty-history and multi-turn runs had no unsupported claims. See [Upgrade validation](upgrade_validation.md) for raw metrics, judge audits and limitations.
