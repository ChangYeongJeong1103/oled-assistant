# AI-Driven OLED Assistant

![GPT-5-mini](https://img.shields.io/badge/Model-GPT--5--mini-10a37f.svg)
![RAG](https://img.shields.io/badge/RAG-Strict_Document--Grounded-green.svg)
![Agent](https://img.shields.io/badge/Agent-Plan_%2B_Tool_Calling-8A2BE2.svg)
![Multi-Agent](https://img.shields.io/badge/Multi--Agent-GPT--6_Luna_%2F_Sol_Routing-8A2BE2.svg)
![Deployment](https://img.shields.io/badge/Deployment-Google_Cloud_Run-blue.svg)
![Status](https://img.shields.io/badge/Status-Production_Ready-success.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An intelligent, secure, and domain-specific RAG (Retrieval-Augmented Generation) assistant for OLED display engineers.

## Overview
This tool allows engineers to ask technical questions about OLED physics, fabrication, and materials. Its **Strict RAG** engine requires relevant document evidence before generation and returns "No Answer" when the corpus cannot support a response.

Retrieved documents are the **primary source**. When they contain partial but relevant evidence, the model may connect missing logical steps using OLED/physics knowledge, but it may not invent unsupported numbers, experimental conditions, or citations.

There are two engines. Both use the same retrieval stack, and `ENGINE_MODE` selects which one runs:
- **`workflow`** (default): The original single-pass Strict RAG pipeline.
- **`agent`**: A tool-calling agent that plans first, can run several searches per turn, searches again with reworded queries when needed, and has to cite the chunks it was shown. On top of it, **multi-model routing** (GPT-6 Luna / GPT-6.1 Sol with one escalation), **orchestrator-worker**, and a **reviewer** can be switched independently with environment flags. The measured default uses all three. All limits (searches, LLM calls, time) are enforced in Python. See [Agent Mode](#agent-mode).

### Key Features
- **Strict Document-Grounded RAG**: The system has no unrestricted LLM fallback. It rejects off-topic questions before generation and returns `NO_ANSWER_IN_DOCS` when retrieved evidence is insufficient.
- **Single-Agent Planning and Re-Search**: In agent mode the model fixes typos and abbreviations, splits multi-part questions, and searches again when the results are weak. A per-request evidence ledger makes sure it can only cite chunks it was actually shown.
- **Multi-Model Routing and Multi-Agent Patterns**: The planner rates how complex a question is, sends easy questions to a light model and hard ones to a heavy model, and allows one escalation in a fresh context. The orchestrator-worker and reviewer share the same ledger and budgets. In our tests, workers improved multi-step retrieval while the reviewer caught answers that only covered related facts, so both are on by default.
- **Verified Source Links**: Answers list the cited paper titles with DOI links that were verified against Crossref (`src/source_registry.json`). Titles and URLs always come from this registry, so the model never writes them itself.
- **Measured Engine Comparison**: `scripts/evaluate_engines.py` runs the workflow and every agent configuration on the same 50 questions and reports false rejections, unsupported claims (graded by an independent judge model), latency, tokens per model, and cost.
- **Public Cloud Demo**: The current application runs on Google Cloud Run and uses **GPT-5-mini** through the OpenAI API. `reasoning_effort="minimal"` keeps document-grounded generation responsive.
- **Privacy-First Internal Track**: The original internal deployment used **Mistral-Nemo via Ollama** on local hardware so sensitive OLED documents did not leave the machine. This remains a separate deployment track, not the runtime used by the public demo in this repository.
- **Production-Style Vector DB Lifecycle**: In cloud deployment, the image ships with a prebuilt `chroma_db` for fast startup. The app only rebuilds from `data/` when the DB is missing or incompatible.
- **Wide Retrieval + BGE Reranking**: Retrieves a wider candidate pool, filters it by a per-document relevance threshold, then uses a `BAAI/bge-reranker-base` cross-encoder to select the strongest final documents for the LLM.
- **Measured, Not Guessed, Thresholds**: `scripts/measure_relevance_distribution.py` dumps the relevance distribution of every retrieved candidate across representative queries, so decision thresholds come from observed separation between on-topic and off-topic questions.
- **Validated Local-Model Track**: Historical experiments found that the optimized Mistral-Nemo internal track produced domain-answer quality comparable to GPT-4o-mini for the evaluated OLED question set.

---

## Screenshots

| RAG Mode - Document-based Answer |
|:---:|
| ![Example RAG](screenshot/Example_RAG.png) |

*The assistant provides detailed, document-grounded answers with relevance scores and response times.*

---

## Key Motivation

### Why Strict RAG?

This tool is designed for **PhD-level domain experts** who need precise, actionable answers—not generic information they could find on Google.

| Standard RAG | Strict RAG (This Tool) |
|--------------|------------------------|
| May answer even when retrieval evidence is weak | Requires at least one document to pass the relevance filter |
| May use broad model knowledge as the answer source | Uses retrieved documents as the primary source and permits only limited logical completion |
| Can produce a plausible answer despite a corpus gap | Returns `NO_ANSWER_IN_DOCS` when the context does not support an answer |

**Why "No Answer" is better than LLM Fallback:**
- If information is not supported by the corpus, the system should expose that coverage gap.
- Returning a "plausible guess" from LLM knowledge would be **dangerous in production** settings.
- Engineers need to know when something is missing so they can **identify it as a next step** in their actual work.
- Generic answers from news/blog or Wikipedia are **not what experts need**—they can Google that themselves.

> **Design Philosophy**: Retrieved technical evidence must exist before the model is allowed to answer. Limited domain reasoning may connect that evidence, but it cannot replace missing evidence.

---

### On-Premise LLM vs Commercial API

This project was originally built and operated as an **Apple-internal, on-premise
OLED assistant**. Sensitive technical documents and inference stayed on local
hardware, with Mistral-Nemo served through Ollama. The internal production data,
vector database, and deployment code cannot be published, so they are not
included in this public repository.

The public repository provides a separate, deployable demonstration of the same
RAG design:

| Track | Current public demo | Internal privacy-first deployment |
|-------|---------------------|-----------------------------------|
| LLM | GPT-5-mini via OpenAI API | Mistral-Nemo via Ollama |
| Runtime | Google Cloud Run | Local/on-premise hardware |
| Data | Publicly deployable OLED corpus and prebuilt ChromaDB | Sensitive internal OLED documents |
| Goal | Reproducible public demonstration | Keep Apple-internal documents and inference fully local |

The current code in `src/rag_engine.py` implements the **public cloud track**
with `ChatOpenAI`. Non-confidential notebooks and evaluation artifacts document
the Mistral-Nemo experiments, but the Apple-internal implementation and data are
intentionally excluded.

An earlier project version directly compared the on-premise Mistral-Nemo system
with the commercial GPT-4o-mini API on the same OLED evaluation set:

| Aspect | Commercial baseline (GPT-4o-mini) | On-premise system (Mistral-Nemo) |
|--------|------------------------------------|----------------------------------|
| Serving | Cloud API | Local Ollama inference |
| Data privacy | Documents leave the local environment when sent as context | Documents and inference remain local |
| Usage cost | Pay per token | No per-token API charge |
| Evaluated answer quality | Strong commercial baseline | Comparable after domain-specific optimization |

This comparison produced an important engineering finding:

> **With proper optimization, on-premise LLMs can achieve commercial-tool-level performance for a constrained domain task.**

**How the local track was optimized:**
1. **Prompt Optimization**: Carefully engineered prompts for domain-specific responses
2. **High-Quality RAG Data**: Curated internal technical documents
3. **Hyperparameter Tuning**: Systematic experiments to find optimal settings (chunk size, relevance threshold, etc.)

In the evaluated OLED question set, PhD-level experts found the optimized Mistral responses comparable in quality to GPT-4o-mini.

> **Note**: This doesn't mean Mistral is "better" than GPT—it means that with the right optimization, you can achieve production-grade results with local, private infrastructure.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31%2B-FF4B4B.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1%2B-1C3C3C.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--5--mini-10a37f.svg)
![Cloud Run](https://img.shields.io/badge/Google_Cloud-Cloud_Run-4285F4.svg)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Deployment-326CE5.svg)
![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

### Architecture Choices
- **App Interface**: `Streamlit` was chosen for rapid prototyping and its native support for chat interfaces (`st.chat_message`).
- **LLM Serving (Current Public App)**: `ChatOpenAI` serves **GPT-5-mini** through the OpenAI API on Google Cloud Run. GPT-5 uses its required temperature (`1.0`) with `reasoning_effort="minimal"`.
- **LLM Serving (Internal Track)**: The privacy-first internal deployment uses **Mistral-Nemo 12B via Ollama** on local hardware; it is documented as a separate deployment path.
- **RAG Orchestration**: Custom retrieval logic performs wide vector search, sigmoid relevance scoring, per-document filtering, and BGE cross-encoder reranking before answer generation.
- **Modular Data Pipeline**: `src/document_pipeline.py` isolates document loading, chunking, embedding, and ChromaDB lifecycle management from `src/rag_engine.py`.
- **Vector Database**: `ChromaDB` is prebuilt into the cloud image for low cold-start latency, then reused at runtime ("rebuild only if missing/incompatible").
- **Container Orchestration**: The same Docker image also runs on Kubernetes (`k8s/oled-assistant.yaml`): a Deployment with resource requests/limits and readiness/liveness probes, a Service for access, and the API key injected from a Kubernetes Secret.

---

## System Architecture

The system follows **Strict RAG** logic with a modern retrieval pipeline: it first retrieves a wider candidate pool, filters weak matches, reranks the survivors, and answers only from the strongest final documents.

```mermaid
graph TD
    User[User Query] --> UI[Streamlit Interface]
    UI --> VectorSearch[Vector Search<br/>Candidate Top-K = 20]
    VectorSearch --> Score[Sigmoid Relevance<br/>per Document]
    Score --> Filter{relevance ≥ 0.50 ?}

    Filter -->|No document survives| Gate{max relevance < 0.25 ?}
    Gate -->|Yes| Reject[🔴 OFF_TOPIC Rejection<br/>no LLM call]
    Gate -->|No| NoAnswer[🟠 NO_ANSWER_IN_DOCS]

    Filter -->|Survivors| Rerank{More than Final Top-N?}
    Rerank -->|Yes| CrossEncoder[BGE Cross-Encoder Reranker]
    Rerank -->|No| FinalDocs[Final Documents]
    CrossEncoder --> FinalDocs

    FinalDocs --> LLM[LLM Generation<br/>returns answer_found flag]
    LLM -->|true| Answer[🟢 Document-Based Answer]
    LLM -->|false| NoAnswer

    Reject --> Final[Final Response]
    Answer --> Final
    NoAnswer --> Final
    Final --> UI

    style Answer fill:#d4edda,stroke:#28a745
    style NoAnswer fill:#fff3cd,stroke:#856404
    style Reject fill:#f8d7da,stroke:#721c24
```

**Key Decision Points:**
- **One score, one scale**: every threshold is compared against `relevance = sigmoid(cosine similarity)`. Raw similarity is only ever the sigmoid's input, because scientific text clusters too tightly in raw space to threshold reliably
- **Candidate Top-K (20)**: Retrieves a wider pool first to reduce missed relevant chunks
- **Document Filter (0.50)**: The one knob controlling answer quality. Documents below it never reach the reranker or the LLM
- **Final Top-N (4)**: Sends only the strongest reranked documents to the LLM
- **Off-Topic Gate (0.25)**: Deliberately loose. It runs only when *no* document survives the filter, and merely picks which rejection the user sees. Measured on this corpus, off-topic queries peak at 0.059 while on-topic queries start at 0.652
- **Grounding Check**: The LLM reports whether the context supported an answer through an explicit `answer_found` flag in its JSON response, so no second scoring call is needed

---

## Agent Mode

The workflow makes every decision from **one** retrieval of the literal question. Because of that, a typo or an informal question can be rejected before the model ever sees a relevant document. `ENGINE_MODE=agent` keeps the same retrieval stack, but lets an agent decide *how* to look for the evidence.

```mermaid
graph TD
    Q[User question] --> Plan["Plan (1 LLM call, JSON schema)<br/>domain + sub-questions"]
    Plan -->|out_of_domain| Off[🔴 OFF_TOPIC]
    Plan -->|in_domain / uncertain| LLM[Agent chooses tools<br/>several searches per turn allowed]
    LLM --> S["search_documents(query)<br/>retrieve → filter ≥ 0.50 → rerank"]
    S --> Ledger[(Evidence ledger<br/>chunks shown in this request)]
    Ledger --> LLM
    LLM --> Sub["submit_answer(answer, citations)"]
    Sub --> V{Every cited ID in the ledger?}
    V -->|no, revisions left| LLM
    V -->|yes| RAG[🟢 RAG + numbered sources]
    LLM --> D["declare_insufficient(missing)"] --> NA[🟠 NO_ANSWER_IN_DOCS]
```

- **Plan First**: The first call returns `in_domain / out_of_domain / uncertain` together with search-ready sub-questions (typos fixed, abbreviations expanded). Only a clear `out_of_domain` ends the run here, so a low score on the first search can never produce `OFF_TOPIC` by itself.
- **Tool Calling**: The model works through `search_documents`, `submit_answer`, and `declare_insufficient`, called via the OpenAI Responses API (function tools with reasoning). It can issue independent searches in the same turn (parallel tool calling), but the finishing tools have to be called alone. If the model sends malformed arguments, an unknown tool, or plain text, we return that to it as an error.
- **Same Relevance Filter**: The model only chooses the query. `MIN_DOC_RELEVANCE`, `CANDIDATE_TOP_K`, and `FINAL_TOP_N` stay in Python, so the agent has no way to bypass the filter.
- **Citation Check**: Each chunk shown to the model gets a stable ID (a hash of file, page, and text). We accept an answer only if every ID it cites was returned in **this** request. Otherwise the errors are sent back so the model can revise.
- **Budgets Enforced in Python**: A single agent gets 3 searches, 6 LLM calls (including planning and corrections), 2 answer revisions, 90 s per question, and 60 s per API call. The budgets grow with each pattern we enable (the default with routing + workers + reviewer has 4 searches, 19 LLM calls, and 150 s), and all agents working on a question share one budget. We check the deadline before every API attempt, around every search, and right before the final answer is accepted. Transient API errors get one retry, but only if it still fits in the time left. Repeated queries are served from a cache. When a limit is reached, the run stops with a recorded reason and we never force an unverified answer.
- **Same Answer Modes**: `RAG`, `NO_ANSWER_IN_DOCS`, and `OFF_TOPIC`, plus `ERROR` for API failures or timeouts. The UI also shows live search progress, numbered sources with verified DOI links, and an **Agent Trace** expander (plan, queries, result counts, stop reason). The same traces are appended to `logs/agent_traces.jsonl`.

### Workflow vs Agent (measured, agent v1 on GPT-5-mini)

Both engines answered the same 34 questions (`eval/questions.json`: normal, typo, paraphrase, abbreviation, ambiguous, multi-hop, no-answer, off-topic). An independent judge model (`gpt-5`) split every answer into claims and graded each claim against the evidence the engine actually used. The agent figures come from two full runs.

| Metric | Workflow | Agent |
| :--- | :---: | :---: |
| Mode accuracy | 88% | 97% |
| False rejection (answerable question rejected) | 15.4% | **0%** |
| Unsupported claims | 13.4% | **5.1 – 6.9%** |
| Key-point coverage | 0.64 | 0.77 – 0.78 |
| Avg LLM calls / searches | 0.7 / 1 | 3.0 – 3.3 / 1.1 |
| Latency mean (p90) | 6.2 s (9.9 s) | 11.0 – 11.3 s (15.1 – 18.7 s) |
| Cost per question (GPT-5-mini) | $0.0008 | $0.0023 – 0.0024 |

- **Recovered Questions**: The workflow rejected a typo ("thermaly activted delayed florescence") and three informal questions ("why do blue OLEDs die so fast?") in 3 of 3 repeated runs. The agent answered them in 3 of 3.
- **Grounding**: Answering more questions did not cost us accuracy. The unsupported-claim rate actually went down.
- **Cost**: About 2× the latency and 3× the token cost, since the agent makes ~3 LLM calls per question instead of 1.
- **Known Weak Spots**: On multi-hop questions the agent sometimes searches only once and then bridges a link that the documents do not state. Raising `reasoning_effort` from `minimal` to `low` did not help (unsupported claims 8.3% → 7.6%, within noise), so we kept `minimal` for GPT-5-mini. Also, in these v1 runs one "no answer" question (supply-chain cost) was answered from a cost-projection table in the corpus, while the label expects a refusal. The answer policy of the GPT-6 version now refuses it (see below).

### Model Routing and Multi-Agent (measured, GPT-6)

The agent now runs on the OpenAI Responses API with **GPT-6 Luna** (light) and **GPT-6.1 Sol** (heavy). We added three patterns on top of the single agent, and each one has its own environment flag:

- **Model Routing**: The planner labels each question `simple` or `complex`. Simple questions go to Luna and complex ones go to Sol. If a Luna run fails in a way that a stronger model could fix, it escalates to Sol **once**. Sol starts in a fresh context with the original chunk text instead of Luna's conversation.
- **Orchestrator-Worker**: Two Luna workers research separate sub-questions, each with its own context and tool loop. They run in parallel, or one after another when the second hop depends on the first. A Sol orchestrator then checks their findings against the **original chunk text** and writes the answer.
- **Reviewer**: Sol checks every claim against the cited chunks, including statements about what the documents do not contain. It does not see the research trace. We accept the answer only if the review is readable, the core question is answered, and no claim is left unsupported. Otherwise the answer goes back to the writer (at most twice), and every revision is reviewed again. If the last review still fails, the user gets `NO_ANSWER_IN_DOCS`.

> **Answer Policy**: We return `RAG` only when the cited evidence answers the **core** of the question, meaning the specific fact, figure, comparison, or mechanism that was asked for. If the agent only found related facts, the result is `NO_ANSWER_IN_DOCS`. There is no partial-answer mode.

All agents working on a question share one evidence ledger, search cache, and budget. Every configuration below answered the same 50 questions and was graded by the same independent judge (`gpt-5`):

| Configuration | False rejection | False acceptance | Unsupported claims | Hard questions: unsupported | Latency mean | Cost / question |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Workflow (gpt-5-mini) | 11.9% | 0% | 13.0% | 9.0% | 7.5 s | $0.0009 |
| Luna only, parallel searches | 2.4% | 12.5% | 4.1% | 7.1% | 19.7 s | $0.0010 |
| Sol only, parallel searches | 0% | 12.5% | 2.6% | 0.9% | 20.1 s | $0.0198 |
| Routing (Luna → Sol) | 0% | 12.5% | 3.8% | 2.8% | 16.8 s | $0.0107 |
| Routing + 2 workers | 0% | 12.5% | 3.2% | 2.1% | 18.2 s | **$0.0057** |
| Routing + reviewer | 2.4% | 12.5% | **2.3%** | 2.0% | 28.1 s | $0.0193 |

False rejection is measured over the 42 answerable questions, and false acceptance over the 8 questions that should be refused. "Hard" means the 13 multi-hop and sequential questions, which are the same in every row. NOTE: these runs were done before the answer-policy fixes described below.

- **False Acceptance (`na1`, supply-chain cost)**: Every agent answered this question instead of refusing it (1 of 8). The answers did not make up cost figures, and their factual claims about the cost of noble metals were supported. The real mistake was answering from related cost remarks, and then saying that "the documents" contain no cost figures after reading only a few chunks. Agents may now only say that the *retrieved evidence* does not state something.

**What we learned:**
- **Routing**: 46% cheaper than using Sol for everything, with similar accuracy. The 27 questions routed to Luna kept its cost and speed (12.6 s, $0.0008).
- **Workers**: A trade-off between Luna and Sol. On the 13 hard questions, routing + workers lowered unsupported claims from 7.1% (Luna only) to 2.1% at less than half of Sol's cost ($0.0116 vs $0.0278). Sol alone was still lower (0.9%), and one small run is not enough to say the team matches it. The cost stays low because the search turns run on Luna, and Sol only reads the condensed evidence and writes the answer.
- **Reviewer**: In this first comparison it had the fewest unsupported claims, but +80% cost and +67% latency. It also caused one false rejection, where it marked a claim as unsupported even though the cited chunk supports it. This was measured before the final reviewer and answer-policy fixes below.
- **Parallel Tool Calling**: Used in 19 of 50 runs, and it covers more ground on multi-part questions. It does not reduce latency here, because retrieval runs on one local CPU.
- **Caveat**: One run per configuration on a small, well-covered corpus, so differences of one or two claims are noise. The original Apple-internal version, with a larger and more complex document set, benefited more from routing and decomposition.

**Final default (routing + 2 workers + reviewer).** We chose this setup based on
repeated tests during development. In those tests the two patterns solved
different problems: the workers made sequential retrieval (`sq2`) more
reliable, while the reviewer consistently refused `na1`, whose evidence only
covered related facts. These tests ran on intermediate versions of the code,
so we treat them as the reason for the choice rather than as a measured
result.

The measured result is the run below. We froze the code (same SHA-256 hashes
before and after the run) and evaluated all 50 questions:

| Mode accuracy | False rejection | False acceptance | Unsupported claims | Partially supported | Key points | Latency mean / p90 | Cost / question |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **98%** (49 / 50) | 2.4% | **0%** | **0.35%** (1 / 289 claims) | 2.77% (8 / 289) | 0.687 | 30.0 / 50.9 s | $0.0183 |

- **Mode error**: The one mode error was `sq2`. This run did not retrieve a chunk
  that directly connects FIrpic, 10 wt%, and the 34.1% EQE device, so the
  reviewer rejected the answer.
- **Hard questions**: Of the 13 hard questions, 12 were answered. Their 95 judged
  claims had 0 unsupported and 4 partially supported.
- **Judge policy**: This run was graded under `scoped_absence_v2`. Under this
  policy, a statement such as "the retrieved evidence does not state X" is
  supported when the evidence indeed lacks X. The table above was graded under
  the earlier `strict_absence_v1` policy, so 0.35% can't be compared with those
  rates as an improvement. To compare two runs, re-grade both under the same
  policy with `scripts/evaluate_engines.py --rejudge`.

For more details, see [Agent Engine](docs/agent_engine.md). Results are in
`eval/results/` (historical comparison: `multi_agent_comparison.md`; final run:
`agent_final_combined_20261004_035751.json`). The published result files keep
chunk IDs, paper titles, and URLs, but not the text of the papers. The full
results stay local in the git-ignored `eval/results/raw/`.

---

## Quick Start

### Option 1: Try Live Demo (Cloud Deployment)

**No installation required!** Access the deployed application directly:

**[https://oled-assistant-961016411722.us-west2.run.app](https://oled-assistant-961016411722.us-west2.run.app)**

> Deployed on Google Cloud Run for instant access.

### Option 2: Run with Docker

The easiest way to run the application with minimal local setup.

```bash
# 1) Build image
docker build -t oled-assistant .

# 2) Run container
docker run -p 8502:8501 -e OPENAI_API_KEY="your-api-key-here" oled-assistant

# Same image, agent engine
docker run -p 8502:8501 -e OPENAI_API_KEY="your-api-key-here" -e ENGINE_MODE=agent oled-assistant

# Agent options (defaults: routing on, 2 workers, reviewer on; each flag is independent)
docker run -p 8502:8501 -e OPENAI_API_KEY="your-api-key-here" -e ENGINE_MODE=agent \
  -e AGENT_REVIEWER=false oled-assistant                       # lower cost: no reviewer
docker run -p 8502:8501 -e OPENAI_API_KEY="your-api-key-here" -e ENGINE_MODE=agent \
  -e AGENT_ROUTING=false -e AGENT_WORKERS=0 -e AGENT_REVIEWER=false \
  oled-assistant                                                # cheapest: Luna only
```

Visit `http://localhost:8502` in your browser.

### Option 3: Run Locally (Code-Only Clone)

1. **Clone the repository**
   ```bash
   git clone https://github.com/ChangYeongJeong1103/oled-assistant.git
   cd oled-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment**
   Create a `.env` file in the root:
   ```env
   OPENAI_API_KEY=sk-...
   ```

4. **Prepare runtime data (required)**
   This GitHub repository is code-only by default (both `data/` and `chroma_db/` are excluded).
   To run locally, provide one of the following:
   - a prebuilt `chroma_db/` folder, or
   - source files (`.pdf` / `.docx`) under `data/` and then build `chroma_db/`.

5. **Run the app**
   ```bash
   streamlit run src/app.py

   # Agent engine
   ENGINE_MODE=agent streamlit run src/app.py
   ```

### Option 4: Run on Kubernetes (Local Cluster)

Runs the same Docker image on a local Kubernetes cluster (tested with Docker Desktop's built-in kubeadm cluster). `k8s/oled-assistant.yaml` defines a Deployment (versioned image tag, `Recreate` update strategy, resource requests/limits, readiness and liveness probes) and a LoadBalancer Service.

**Prerequisites**: Docker Desktop with Kubernetes enabled (Settings → Kubernetes), about 8 GB of memory allocated to Docker, and a prebuilt `chroma_db/` folder (see Option 3, step 4).

1. **Create a key-only env file** named `.env.k8s` in the root (git-ignored). Keep only the key the app needs, because every line becomes part of the Secret:
   ```env
   OPENAI_API_KEY=sk-...
   ```

2. **Build, store the key, and deploy**
   ```bash
   # 1) Build image (the tag must match the image name in the manifest)
   docker build -t oled-assistant:v1 .

   # 2) Store the API key as a Kubernetes Secret (never written into the manifest)
   kubectl create secret generic openai-secret --from-env-file=.env.k8s

   # 3) Deploy and wait until the pod is ready
   kubectl apply -f k8s/oled-assistant.yaml
   kubectl rollout status deploy/oled-assistant
   ```

3. **Open** `http://localhost:8080` in your browser.

**Operations**
```bash
# Stop / start the app without deleting its configuration
kubectl scale deploy/oled-assistant --replicas=0
kubectl scale deploy/oled-assistant --replicas=1

# Release a new version: build a new tag, update "image:" in the manifest, then apply
docker build -t oled-assistant:v2 .
kubectl apply -f k8s/oled-assistant.yaml

# Roll back to the previous version (the old image tag must still exist)
kubectl rollout undo deploy/oled-assistant
kubectl rollout history deploy/oled-assistant

# Remove everything
kubectl delete -f k8s/oled-assistant.yaml
kubectl delete secret openai-secret
```

**Known limitations of this local setup**
- **Readiness**: The probe checks that the Streamlit server responds, not that the embedding model, reranker, and ChromaDB are loaded.
- **First question**: Models load on first use, so the first response is slower than later ones.
- **Session**: Chat history lives in the pod's memory and is lost when the pod is replaced.
- **Downtime on update**: `Recreate` stops the old pod before starting the new one (about 20 s of downtime), because two 4Gi pods do not fit on an ~8 GB single node.

**Verified on Docker Desktop (single node, ~8 GB)**
- All three modes are unchanged: OLED questions return `RAG`, a cooking question returns `OFF_TOPIC` without an LLM call, and questions about patents or supply-chain cost return `NO_ANSWER_IN_DOCS`.
- Self-healing: after deleting the pod, the Deployment created a replacement within 1 s and the app was healthy again in 22 s.
- Update and rollback: `v1 → v2` through the manifest and `v2 → v1` through `kubectl rollout undo` each finished in about 23 s, confirmed by the running image ID.
- Latency: model and vector DB loading took about 10 s; the first question took about 12.5 s and later questions about 3.5–9 s.

> With `kind` or `minikube`, load the local image into the cluster first (e.g. `kind load docker-image oled-assistant:v1`).

---

## Project Structure

```text
oled-assistant/
├── src/                  # Source Code
│   ├── __init__.py       # Package marker
│   ├── app.py            # Main Streamlit Application
│   ├── rag_engine.py     # Strict RAG Logic Class (workflow engine)
│   ├── agent_runtime.py  # Agent engine: planning, routing, tool loop, budgets, trace
│   ├── agent_team.py     # Orchestrator-worker and reviewer
│   ├── agent_prompts.py  # Prompts and tool schemas for every agent role
│   ├── agent_tools.py    # search_documents tool, evidence ledger, citation check
│   ├── source_registry.py   # File name -> paper title / verified DOI link
│   ├── source_registry.json # Generated by scripts/build_source_registry.py
│   ├── document_pipeline.py # Document loading/chunking/vector DB lifecycle
│   ├── config.py         # Configuration & Hyperparameters
│   └── utils.py          # Logging & Helper Functions
├── scripts/              # Tuning & Validation Utilities
│   ├── measure_relevance_distribution.py  # Relevance distribution measurement
│   ├── evaluate_engines.py                # Engine / agent-configuration evaluation (judge, latency, cost)
│   ├── probe_corpus.py                    # Show what the search tool returns (grounding eval questions)
│   └── build_source_registry.py           # Crossref-verified title/DOI registry
├── eval/                 # Evaluation set and results
│   ├── questions.json    # 50 questions (normal/typo/.../easy/multi-hop/sequential/off-topic)
│   └── results/          # Per-run JSON results (no chunk text) and comparison tables;
│                         # raw/ keeps full results locally (git-ignored)
├── data/                 # Optional local-only source docs for rebuilding vector DB
├── notebooks/            # Development Notebooks
│   ├── OLED_assistant_v1_HP_tuning.ipynb  # Hyperparameter tuning
│   ├── OLED_assistant_v2_Mistral.ipynb    # Mistral integration
│   ├── OLED_assistant_v3_final.ipynb      # Strict RAG baseline
│   ├── OLED_assistant_v4_sim.ipynb        # Simulation and validation notebook
│   ├── OLED_assistant_v5_updated.ipynb    # Updated modular validation notebook
│   └── OLED_Assistant_v6_GCP.ipynb        # Cloud deployment validation notebook
├── chroma_db/            # Prebuilt persistent vector DB (cloud image includes this folder)
├── docs/                 # Documentation & Experiments
│   ├── architecture.md   # System Flowchart
│   ├── rag_engine.md     # Logic Explanation (workflow)
│   ├── agent_engine.md   # Agent design, limits, evaluation
│   ├── hyperparameter.md # Hyperparameter Tuning Guide
│   ├── llm_comparison.md # LLM Comparison Results
│   └── experiments/      # Research Data (logs, CSVs)
├── k8s/                  # Kubernetes Manifests
│   └── oled-assistant.yaml  # Deployment + Service
├── logs/                 # Usage Logs
├── screenshot/           # Demo Screenshots
├── requirements.txt      # Python Dependencies
└── README.md             # This file
```

## Documentation
- [System Architecture](docs/architecture.md)
- [RAG Engine Logic](docs/rag_engine.md)
- [Hyperparameter Tuning](docs/hyperparameter.md)
- [LLM Comparison](docs/llm_comparison.md)
- [Agent Engine](docs/agent_engine.md)

## Future Work

### LLM Fine-Tuning (Next Phase)

- **Goal**: Build an OLED-specialized Mistral that natively understands domain terminology
- **Expected Benefits**: Faster responses, better consistency, reduced prompt complexity
- **Approach**: QLoRA fine-tuning with 500+ expert-validated Q&A pairs from internal documents

---
**Developed by CYJ for XF Team**
