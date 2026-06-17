"""
Strict RAG Engine for OLED Assistant
Aligned with notebooks/OLED_assistant_v3_final.ipynb
"""
import math

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
        relevance_threshold,
        candidate_top_k,
        min_document_similarity,
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
        self.relevance_threshold = relevance_threshold
        self.candidate_top_k = candidate_top_k
        self.min_document_similarity = min_document_similarity
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

        # Strict RAG prompt
        rag_prompt_template = """You are an OLED Display Technical Assistant.
Answer the question using the provided context documents as your PRIMARY source.

RULES:
1. Always read the Context carefully and base your answer as much as possible on the Context.
2. If the Context contains partial but relevant information, you MAY use your own OLED/physics knowledge to fill in missing logical steps.
3. ONLY when the Context is clearly irrelevant or provides almost no signal, say: "Information not found in the provided OLED documents."
4. Never contradict the facts given in the Context.
5. Do NOT hallucinate specific numbers, experimental conditions, or paper titles that are not supported by the Context.

Context: {context}

Question: {question}

Answer:"""
        
        self.rag_prompt = PromptTemplate(
            template=rag_prompt_template,
            input_variables=["context", "question"]
        )

    @staticmethod
    def _distance_to_similarity(distance) -> float:
        """
        Convert Chroma's L2 distance into a similarity score.

        The BGE embeddings are stored with normalize_embeddings=True, so this
        conversion approximates cosine similarity and keeps scores in [0, 1].
        """
        sim = 1.0 - (float(distance) * float(distance)) / 2.0
        return max(0.0, min(1.0, sim))

    def retrieve_candidates(self, query):
        """Retrieve a wide candidate pool and attach similarity scores."""
        docs_with_scores = self.vectorstore.similarity_search_with_score(
            query,
            k=self.candidate_top_k,
        )
        if not docs_with_scores:
            return []

        candidates = []
        for doc, distance in docs_with_scores:
            candidates.append({
                "doc": doc,
                "distance": float(distance),
                "similarity": self._distance_to_similarity(distance),
            })

        return candidates

    def get_relevance_score(self, candidates):
        """Calculate query relevance from the strongest retrieved candidates."""
        if not candidates:
            return 0.0

        # Keep the old intuition: the final gate is based on the strongest few
        # documents, not every low-signal candidate in the wider pool.
        top_candidates = candidates[:self.final_top_n]
        scores = [candidate["similarity"] for candidate in top_candidates]
        avg_score = sum(scores) / len(scores)

        sigmoid_score = 1 / (1 + math.exp(-self.sigmoid_steepness * (avg_score - self.sigmoid_midpoint)))
        return sigmoid_score

    def filter_candidates_by_similarity(self, candidates):
        """Keep only documents that meet the minimum similarity threshold."""
        return [
            candidate
            for candidate in candidates
            if candidate["similarity"] >= self.min_document_similarity
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
                key=lambda candidate: candidate["similarity"],
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
                "Reranking failed. Falling back to similarity ranking: %s",
                str(exc),
            )
            sorted_candidates = sorted(
                candidates,
                key=lambda candidate: candidate["similarity"],
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
            source_label = source if page is None else f"{source} (page {page + 1})"

            context_parts.append(
                f"[Document {index}]\n"
                f"Source: {source_label}\n"
                f"Similarity: {candidate['similarity']:.3f}\n"
                f"Content:\n{doc.page_content}"
            )

        return "\n\n---\n\n".join(context_parts)

    def generate_answer(self, question, final_candidates):
        """Generate an answer from the reranked final context."""
        context = self.build_context(final_candidates)
        prompt = self.rag_prompt.format(context=context, question=question)
        response = self.llm.invoke(prompt)
        return getattr(response, "content", str(response))

    def query(self, question):
        """Process query through Strict RAG logic."""
        candidates = self.retrieve_candidates(question)
        relevance_score = self.get_relevance_score(candidates)
        filtered_candidates = self.filter_candidates_by_similarity(candidates)
        
        result = {
            "answer": None,
            "mode": None,
            "relevance_score": relevance_score,
            "retrieved_docs": [],
            "retrieval_metadata": {
                "candidate_top_k": self.candidate_top_k,
                "candidate_count": len(candidates),
                "similarity_threshold": self.min_document_similarity,
                "filtered_count": len(filtered_candidates),
                "final_top_n": self.final_top_n,
                "final_doc_count": 0,
                "reranker_used": False,
            },
        }
        
        # Check relevance threshold
        if relevance_score >= self.relevance_threshold and filtered_candidates:
            logger.info(f"✅ High relevance ({relevance_score:.3f}). Executing RAG.")
            result["mode"] = "RAG"

            final_candidates, reranker_used = self.rerank_candidates(
                question,
                filtered_candidates,
            )
            result["retrieval_metadata"]["reranker_used"] = reranker_used
            result["retrieval_metadata"]["final_doc_count"] = len(final_candidates)
            result["retrieved_docs"] = [
                candidate["doc"]
                for candidate in final_candidates
            ]

            try:
                rag_response = self.generate_answer(question, final_candidates)
                result["answer"] = rag_response

                # Check for "Information not found" response from LLM
                if "Information not found" in rag_response or ("provided context" in rag_response and "does not contain" in rag_response):
                    result["mode"] = "NO_ANSWER_IN_DOCS"
                    result["answer"] = "No Answer: The relevant content is not found in RAG documents."
                    logger.info("❌ Documents found but LLM could not find answer in context.")

            except Exception as e:
                logger.error(f"RAG generation failed: {str(e)}")
                result["answer"] = "Error processing request."
                
        else:
            logger.info(f"🚫 Low relevance ({relevance_score:.3f}). Rejecting.")
            result["mode"] = "OFF_TOPIC"
            result["answer"] = "No Answer: The question is not related to OLED display or relevant documents are not available."
            
        return result
