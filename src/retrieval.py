"""Retrieval stack for the OLED Assistant.

Every `search_documents` call retrieves a wide candidate pool, applies the relevance filter, and reranks the survivors with a cross-encoder.
All thresholds live in Python, so the agent chooses only the query and cannot lower a threshold or skip filtering.
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
    """Reuse an existing ChromaDB or build one from source documents when it is missing.

    An existing database that cannot be opened is preserved and raises an error instead of being rebuilt automatically.
    """
    return get_or_create_vectorstore(
        embeddings=embeddings,
        docs_folder=config.DOCS_FOLDER,
        persist_directory=config.DB_PATH,
    )


def build_retriever():
    """Create the `Retriever` with settings from config.

    The application, evaluator and corpus probe all use this function so they search in the same way.
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
    """Perform wide vector search, relevance filtering and cross-encoder reranking.

    The instance owns the vector store, reranker and every retrieval threshold.
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
        """Initialize retrieval and optional reranking."""
        self.vectorstore = vectorstore
        self.candidate_top_k = candidate_top_k
        self.min_doc_relevance = min_doc_relevance
        self.final_top_n = final_top_n
        self.sigmoid_midpoint = sigmoid_midpoint
        self.sigmoid_steepness = sigmoid_steepness

        # The cross-encoder scores each query-document pair more precisely than embedding similarity.
        # It runs only when enough documents survive to require a top-N choice.
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
        """Convert Chroma's squared L2 distance into relevance.

        Chroma's `l2` space returns squared Euclidean distance.
        BGE embeddings are normalized, so `d = ||q - e||² = 2 - 2*cos` and therefore `cos = 1 - d / 2`.
        The raw cosine similarity is then converted to relevance with the configured sigmoid.
        Scientific-document similarities occupy a narrow range, so the sigmoid provides a more useful threshold scale.
        Every retrieval threshold compares against the returned relevance rather than raw similarity.
        """
        similarity = 1.0 - float(distance) / 2.0
        similarity = max(0.0, min(1.0, similarity))

        exponent = -self.sigmoid_steepness * (similarity - self.sigmoid_midpoint)
        return 1.0 / (1.0 + math.exp(exponent))

    def retrieve_candidates(self, query):
        """Retrieve a wide candidate pool and attach relevance scores.

        Chroma searches by L2 distance.
        Because the similarity and sigmoid transformations are monotonic, top-k by distance is also top-k by relevance.
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
        """Keep only candidates that clear the document relevance threshold.

        This threshold controls context quality.
        Documents below it never reach the cross-encoder or the LLM.
        """
        return [
            candidate
            for candidate in candidates
            if candidate["relevance"] >= self.min_doc_relevance
        ]

    def rerank_candidates(self, query, candidates):
        """Rerank surviving documents with the cross-encoder.

        Reranking is skipped when `final_top_n` or fewer candidates survive because all of them will be returned.
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
