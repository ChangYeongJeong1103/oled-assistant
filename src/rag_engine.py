"""
Strict RAG Engine for OLED Assistant
Aligned with notebooks/OLED_assistant_v3_final.ipynb
"""
import json
import math
import os

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from sentence_transformers import CrossEncoder

import config
from document_pipeline import create_embeddings_model, get_or_create_vectorstore
from utils import logger


def create_llm(model_name: str, temperature: float):
    """
    Create OpenAI-compatible chat model for cloud deployment.

    Args:
        model_name: OpenAI model name (e.g., "gpt-5-mini", "gpt-4o-mini")
        temperature: Response diversity (0.0 = deterministic, 1.0 = creative)

    Returns:
        ChatOpenAI: LLM instance using OPENAI_API_KEY from environment

    Note:
        GPT-5 family models (gpt-5, gpt-5-mini, gpt-5-nano, ...) only accept the
        default temperature (1.0). They are also "reasoning" models that think
        before answering, which makes them slow by default. For a document
        grounded RAG task we don't need deep reasoning, so we lower the
        reasoning effort to "minimal" to get near gpt-4o-mini latency while
        keeping the newer model's stronger instruction-following.
    """
    # Detect GPT-5 family models, which behave differently from older models.
    is_gpt5_family = model_name.lower().startswith("gpt-5")

    if is_gpt5_family:
        # 1) Pin temperature to the ONLY value GPT-5 models allow (1.0).
        #    We set it explicitly (instead of omitting it) because some
        #    langchain-openai versions auto-send their own default temperature
        #    (e.g., 0.7) when it is not provided, which GPT-5 would reject.
        # 2) reasoning_effort="minimal" tells GPT-5 to skip most of its hidden
        #    "thinking" step. This is the single biggest latency win for RAG.
        #    Passed via model_kwargs so it works on the pinned langchain-openai
        #    version, which has no dedicated reasoning_effort argument.
        return ChatOpenAI(
            model=model_name,
            temperature=1.0,
            model_kwargs={"reasoning_effort": "minimal"},
        )

    # Older models (e.g., gpt-4o-mini) still support a custom temperature.
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
    )


def create_embeddings():
    try:
        # Keep this wrapper for backward compatibility with app.py import.
        return create_embeddings_model()
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {str(e)}")
        raise


def get_vectorstore(embeddings):
    """
    Keep a stable interface while delegating lifecycle logic to document_pipeline.
    """
    return get_or_create_vectorstore(
        embeddings=embeddings,
        docs_folder=config.DOCS_FOLDER,
        persist_directory=config.DB_PATH,
    )


class StrictRAGAssistant:
    """
    Strict RAG System: Answers questions ONLY based on provided documents.
    Returns 'No Answer' if information is missing or question is off-topic.
    """
    
    def __init__(
        self,
        vectorstore,
        llm_model,
        off_topic_threshold,
        candidate_top_k,
        min_doc_relevance,
        final_top_n,
        reranker_enabled,
        reranker_model,
        temperature,
        sigmoid_midpoint,
        sigmoid_steepness,
    ):
        """
        Initialize the retrieval, reranking, and generation components.
        """
        self.vectorstore = vectorstore
        self.off_topic_threshold = off_topic_threshold
        self.candidate_top_k = candidate_top_k
        self.min_doc_relevance = min_doc_relevance
        self.final_top_n = final_top_n
        self.sigmoid_midpoint = sigmoid_midpoint
        self.sigmoid_steepness = sigmoid_steepness

        # Create cloud LLM (OpenAI API)
        self.llm = create_llm(model_name=llm_model, temperature=temperature)

        # Cross-encoder reranker scores each (query, document) pair more
        # precisely than embedding similarity. It runs only when enough
        # documents pass the similarity filter.
        self.reranker = None
        if reranker_enabled:
            try:
                self.reranker = CrossEncoder(reranker_model)
                logger.info("Loaded reranker model: %s", reranker_model)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to load reranker model '%s'. Falling back to similarity ranking: %s",
                    reranker_model,
                    str(exc),
                )

        # Strict RAG prompt.
        #
        # The model reports whether the context actually supported an answer
        # through an explicit `answer_found` flag instead of a magic phrase.
        # Detecting refusals by string matching used to break whenever the
        # model reworded itself ("I couldn't find...", "The documents omit...").
        # The flag costs no extra LLM call, so latency is unchanged.
        rag_prompt_template = """You are an OLED Display Technical Assistant.
Answer the question using the provided context documents as your PRIMARY source.

RULES:
1. Always read the Context carefully and base your answer as much as possible on the Context.
2. If the Context contains partial but relevant information, you MAY use your own OLED/physics knowledge to fill in missing logical steps.
3. Never contradict the facts given in the Context.
4. Do NOT hallucinate specific numbers, experimental conditions, or paper titles that are not supported by the Context.
5. Set "answer_found" to false ONLY when the Context is clearly irrelevant or
   gives almost no signal about the Question. In that case return an empty
   string for "answer".

Respond with ONLY a JSON object in exactly this format, and no other text:
{{"answer_found": true, "answer": "your technical answer here"}}

Context: {context}

Question: {question}

JSON:"""
        
        self.rag_prompt = PromptTemplate(
            template=rag_prompt_template,
            input_variables=["context", "question"]
        )

    def distance_to_relevance(self, distance) -> float:
        """
        Convert Chroma's L2 distance into a RELEVANCE score.

        Two steps happen here, and the intermediate value never leaves this
        method on purpose:

        1. distance -> cosine similarity. The BGE embeddings are stored with
           normalize_embeddings=True, so for unit vectors d^2 = 2 - 2*cos,
           which rearranges to cos = 1 - d^2 / 2.
        2. cosine similarity -> relevance, via the sigmoid. Raw similarities in
           a scientific corpus sit in a narrow band and are useless as a
           decision axis; the sigmoid spreads that band out.

        Every threshold in this engine compares against the value returned
        here, never against the raw similarity.
        """
        similarity = 1.0 - (float(distance) * float(distance)) / 2.0
        similarity = max(0.0, min(1.0, similarity))

        exponent = -self.sigmoid_steepness * (similarity - self.sigmoid_midpoint)
        return 1.0 / (1.0 + math.exp(exponent))

    def retrieve_candidates(self, query):
        """
        Retrieve a wide candidate pool and attach relevance scores.

        Chroma searches its index on L2 distance, which we cannot change. That
        is fine: the sigmoid is monotonic in similarity and similarity is
        monotonic in distance, so the top-k by distance is exactly the top-k by
        relevance.
        """
        docs_with_scores = self.vectorstore.similarity_search_with_score(
            query,
            k=self.candidate_top_k,
        )

        return [
            {"doc": doc, "relevance": self.distance_to_relevance(distance)}
            for doc, distance in docs_with_scores
        ]

    def filter_candidates_by_relevance(self, candidates):
        """
        Keep only documents good enough to be worth reranking.

        This is the single knob that controls context quality. Documents below
        the bar never reach the cross-encoder or the LLM.
        """
        return [
            candidate
            for candidate in candidates
            if candidate["relevance"] >= self.min_doc_relevance
        ]

    def rerank_candidates(self, query, candidates):
        """
        Rerank candidate documents with a cross-encoder.

        If there are only FINAL_TOP_N or fewer candidates, reranking is skipped
        because all surviving documents will be sent to the LLM anyway.
        """
        if len(candidates) <= self.final_top_n:
            return candidates, False

        if self.reranker is None:
            sorted_candidates = sorted(
                candidates,
                key=lambda candidate: candidate["relevance"],
                reverse=True,
            )
            return sorted_candidates[:self.final_top_n], False

        try:
            pairs = [
                (query, candidate["doc"].page_content)
                for candidate in candidates
            ]
            reranker_scores = self.reranker.predict(pairs)

            for candidate, score in zip(candidates, reranker_scores):
                candidate["reranker_score"] = float(score)

            reranked_candidates = sorted(
                candidates,
                key=lambda candidate: candidate["reranker_score"],
                reverse=True,
            )
            return reranked_candidates[:self.final_top_n], True

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Reranking failed. Falling back to relevance ranking: %s",
                str(exc),
            )
            sorted_candidates = sorted(
                candidates,
                key=lambda candidate: candidate["relevance"],
                reverse=True,
            )
            return sorted_candidates[:self.final_top_n], False

    @staticmethod
    def build_context(final_candidates):
        """Build the text context that will be passed to the LLM."""
        context_parts = []
        for index, candidate in enumerate(final_candidates, 1):
            doc = candidate["doc"]
            source = doc.metadata.get("source", "Unknown source")
            page = doc.metadata.get("page")
            # Keep prompt provenance useful without sending local/container paths
            # to the LLM provider.
            source_label = os.path.basename(source) if source else "Unknown source"
            if page is not None:
                source_label = f"{source_label} (page {page + 1})"

            context_parts.append(
                f"[Document {index}]\n"
                f"Source: {source_label}\n"
                f"Relevance: {candidate['relevance']:.3f}\n"
                f"Content:\n{doc.page_content}"
            )

        return "\n\n---\n\n".join(context_parts)

    def generate_answer(self, question, final_candidates):
        """Generate an answer from the reranked final context."""
        context = self.build_context(final_candidates)
        prompt = self.rag_prompt.format(context=context, question=question)
        response = self.llm.invoke(prompt)
        return getattr(response, "content", str(response))

    @staticmethod
    def parse_answer_payload(raw_response):
        """
        Read the JSON envelope the model was asked to produce.

        Returns a (answer_found, answer_text) tuple.

        The model sometimes wraps its JSON in prose or a markdown fence, so we
        slice from the first '{' to the last '}' before parsing. When parsing
        fails we treat the whole response as a normal answer: a formatting slip
        should degrade to "answered" rather than throw away a good answer.
        """
        start = raw_response.find("{")
        end = raw_response.rfind("}") + 1

        if start != -1 and end > start:
            try:
                payload = json.loads(raw_response[start:end])
                answer_found = bool(payload.get("answer_found", True))
                answer_text = str(payload.get("answer", "")).strip()

                if not answer_found:
                    return False, ""
                if answer_text:
                    return True, answer_text
            except (json.JSONDecodeError, AttributeError, TypeError):
                logger.warning("Could not parse the JSON answer envelope; using raw text.")

        return True, raw_response.strip()

    def query(self, question):
        """
        Process a query through Strict RAG logic.

        Decision order:
          1. Retrieve a wide pool and score every candidate as relevance.
          2. Keep the documents that clear MIN_DOC_RELEVANCE.
          3. If none survive, nothing can be grounded. OFF_TOPIC_THRESHOLD then
             only picks which rejection message fits.
          4. Otherwise rerank the survivors and let the LLM answer, which may
             still report that the context did not cover the question.
        """
        candidates = self.retrieve_candidates(question)
        survivors = self.filter_candidates_by_relevance(candidates)

        # Relevance of the single best document. Used for the OFF_TOPIC vs
        # NO_ANSWER_IN_DOCS distinction and surfaced in the UI.
        max_relevance = max(
            (candidate["relevance"] for candidate in candidates),
            default=0.0,
        )

        result = {
            "answer": None,
            "mode": None,
            "relevance_score": max_relevance,
            "retrieved_docs": [],
            "retrieval_metadata": {
                "candidate_top_k": self.candidate_top_k,
                "candidate_count": len(candidates),
                "max_relevance": round(max_relevance, 3),
                "min_doc_relevance": self.min_doc_relevance,
                "survivor_count": len(survivors),
                "final_top_n": self.final_top_n,
                "final_doc_count": 0,
                "reranker_used": False,
            },
        }

        if not survivors:
            # No document is good enough to ground an answer, so the only
            # question left is which rejection the user deserves to see.
            if max_relevance < self.off_topic_threshold:
                result["mode"] = "OFF_TOPIC"
                result["answer"] = (
                    "No Answer: The question is not related to OLED display technology."
                )
                logger.info("🚫 Off-topic (max relevance %.3f). Rejecting.", max_relevance)
            else:
                result["mode"] = "NO_ANSWER_IN_DOCS"
                result["answer"] = (
                    "No Answer: The relevant content is not found in RAG documents."
                )
                logger.info(
                    "🟠 On-topic (max relevance %.3f) but no document cleared the filter.",
                    max_relevance,
                )
            return result

        final_candidates, reranker_used = self.rerank_candidates(question, survivors)
        result["retrieval_metadata"]["reranker_used"] = reranker_used
        result["retrieval_metadata"]["final_doc_count"] = len(final_candidates)
        result["retrieved_docs"] = [candidate["doc"] for candidate in final_candidates]

        try:
            raw_response = self.generate_answer(question, final_candidates)
            answer_found, answer_text = self.parse_answer_payload(raw_response)

            if answer_found:
                result["mode"] = "RAG"
                result["answer"] = answer_text
                logger.info(
                    "✅ Answered from %d documents (max relevance %.3f).",
                    len(final_candidates),
                    max_relevance,
                )
            else:
                result["mode"] = "NO_ANSWER_IN_DOCS"
                result["answer"] = (
                    "No Answer: The relevant content is not found in RAG documents."
                )
                logger.info("❌ Documents retrieved but the LLM found no answer in them.")

        except Exception as exc:  # noqa: BLE001
            logger.error("RAG generation failed: %s", str(exc))
            result["mode"] = "ERROR"
            result["answer"] = "Error processing request."

        return result
