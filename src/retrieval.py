"""
Retrieval stack for the OLED Assistant.

This is the "R" in RAG. The agent's search_documents tool (agent_tools.py) is
the only caller, and every search runs the same three steps:

    retrieve a wide candidate pool -> relevance filter -> cross-encoder rerank

All thresholds live here and in config, so the agent only chooses the query
text and has no way to lower a threshold or skip the filter.
"""
import math

from sentence_transformers import CrossEncoder

import config
from document_pipeline import create_embeddings_model, get_or_create_vectorstore
from utils import logger


def create_embeddings():
    """Load the embedding model (the same one that built the ChromaDB)."""
    try:
        return create_embeddings_model()
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {str(e)}")
        raise


def get_vectorstore(embeddings):
    """
    Open the persisted ChromaDB. document_pipeline rebuilds it from data/ only
    when it is missing or incompatible.
    """
    return get_or_create_vectorstore(
        embeddings=embeddings,
        docs_folder=config.DOCS_FOLDER,
        persist_directory=config.DB_PATH,
    )


def build_retriever():
    """
    Create the Retriever with the settings in config.

    The app, the evaluator, and probe_corpus.py all build it through this
    function, so they always search exactly the same way.
    """
    return Retriever(
        vectorstore=get_vectorstore(create_embeddings()),
        candidate_top_k=config.CANDIDATE_TOP_K,
        min_doc_relevance=config.MIN_DOC_RELEVANCE,
        final_top_n=config.FINAL_TOP_N,
        reranker_enabled=config.RERANKER_ENABLED,
        reranker_model=config.RERANKER_MODEL,
        sigmoid_midpoint=config.SIGMOID_MIDPOINT,
        sigmoid_steepness=config.SIGMOID_STEEPNESS,
    )


class Retriever:
    """
    Wide vector search, relevance filtering, and cross-encoder reranking.

    It owns the vector store, the reranker, and every retrieval threshold.
    """

    def __init__(
        self,
        vectorstore,
        candidate_top_k,
        min_doc_relevance,
        final_top_n,
        reranker_enabled,
        reranker_model,
        sigmoid_midpoint,
        sigmoid_steepness,
    ):
        """
        Initialize the retrieval and reranking components.
        """
        self.vectorstore = vectorstore
        self.candidate_top_k = candidate_top_k
        self.min_doc_relevance = min_doc_relevance
        self.final_top_n = final_top_n
        self.sigmoid_midpoint = sigmoid_midpoint
        self.sigmoid_steepness = sigmoid_steepness

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

    def distance_to_relevance(self, distance) -> float:
        """
        Convert Chroma's L2 distance into a RELEVANCE score.

        Two steps happen here, and the intermediate value never leaves this
        method on purpose:

        1. distance -> cosine similarity. Chroma's "l2" space gives us the
           SQUARED Euclidean distance d = ||q - e||^2. Our BGE embeddings are
           stored with normalize_embeddings=True, so for unit vectors
           d = 2 - 2*cos, which rearranges to cos = 1 - d / 2.
        2. cosine similarity -> relevance, via the sigmoid. Raw similarities in
           a scientific corpus sit in a narrow band and are useless as a
           decision axis; the sigmoid spreads that band out.

        Every threshold in this project compares against the value returned
        here, never against the raw similarity.
        """
        similarity = 1.0 - float(distance) / 2.0
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
        because all surviving documents will be returned anyway.
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
