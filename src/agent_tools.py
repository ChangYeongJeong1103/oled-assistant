"""Search tools and per-request evidence tracking for the OLED agent.

The agent cannot access the vector store directly.
Its only search path is `search_documents(query)`, which retrieves a wide candidate pool, applies the relevance filter, reranks the survivors, and returns the final chunks.
The model chooses search queries but cannot access the vector store or change retrieval settings.
The relevance threshold, candidate-pool size and final top-N all come from the `Retriever`.
Returned chunks enter an `EvidenceLedger`, and final citations must reference chunks shown during the same request.
"""

import hashlib
import os
import re
import threading

from source_registry import describe_source

# Chunk IDs use `c` plus the first eight SHA-1 characters.
# Eight characters keep collisions negligible for roughly 7,000 chunks while remaining easy for the model to copy.
CHUNK_ID_PATTERN = re.compile(r"\bc[0-9a-f]{8}\b")
# Anything that looks like an attempt at a chunk ID, including malformed ones.
CHUNK_ID_LIKE_PATTERN = re.compile(r"\bc[0-9a-f]{5,12}\b")
# Match one or more chunk IDs inside square brackets.
INLINE_CITATION_PATTERN = re.compile(r"\[\s*(c[0-9a-f]{8}(?:\s*[,;]\s*c[0-9a-f]{8})*)\s*\]")


class AgentStop(Exception):
    """Raised from anywhere inside a run to end it with the given stop reason."""

    def __init__(self, reason, detail=""):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def make_chunk_id(doc) -> str:
    """Create a stable chunk ID from its file name, page and text.

    LangChain results from the persisted ChromaDB do not include a stable ID.
    Hashing the fields that identify a chunk gives the same ID across searches and requests.
    """
    metadata = doc.metadata or {}
    key = "|".join(
        [
            os.path.basename(metadata.get("source", "") or ""),
            str(metadata.get("page")),
            doc.page_content,
        ]
    )
    return "c" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]


def normalize_query(query: str) -> str:
    """Normalize only whitespace when creating a search-cache key.

    Signs, decimal points, case and punctuation are preserved because they can change meaning.
    For example, `-5 V` differs from `+5 V`, `1.5 eV` differs from `15 eV`, and `CO` differs from `Co`.
    """
    return " ".join(query.split())


# ================================
# Evidence ledger
# ================================
class EvidenceLedger:
    """Store every chunk shown to any agent during one user request.

    Insertion order keeps source display stable.
    Workers share the ledger across threads, so writes are protected by a lock.
    """

    def __init__(self):
        self._entries = {}
        self._lock = threading.Lock()

    def add(self, chunk_id: str, entry: dict) -> bool:
        """Store a chunk. Returns True if it is new, False if we have already seen it."""
        with self._lock:
            if chunk_id in self._entries:
                return False
            self._entries[chunk_id] = entry
            return True

    def __contains__(self, chunk_id: str) -> bool:
        return chunk_id in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, chunk_id: str) -> dict:
        return self._entries[chunk_id]

    def entries(self) -> list:
        """Return a snapshot of every stored chunk, in insertion order."""
        with self._lock:
            return list(self._entries.values())


# ================================
# search_documents tool
# ================================
class SearchTool:
    """Wrap the Retriever as the agent's only search tool."""

    def __init__(self, retriever):
        # Retriever owns the vector store, reranker and every threshold.
        self.retriever = retriever
        # Embedding and reranking share one CPU model instance.
        # Concurrent searches therefore run one at a time behind this lock.
        self._lock = threading.Lock()

    def search(self, query: str, ledger: EvidenceLedger) -> dict:
        """Retrieve, filter and rerank one query.

        Every returned chunk is added to the request ledger.
        The response contains the status, maximum candidate relevance, filter threshold and final chunk records.
        Maximum relevance is retained even when no candidate survives so the agent can decide whether a rewritten search may help.
        """
        with self._lock:
            return self._search(query, ledger)

    def _search(self, query, ledger):
        candidates = self.retriever.retrieve_candidates(query)
        survivors = self.retriever.filter_candidates_by_relevance(candidates)
        max_relevance = max((c["relevance"] for c in candidates), default=0.0)

        response = {
            "status": "ok" if survivors else "no_relevant_documents",
            "query": query,
            "max_relevance": round(max_relevance, 3),
            "min_doc_relevance": self.retriever.min_doc_relevance,
            "candidate_count": len(candidates),
            "survivor_count": len(survivors),
            "reranker_used": False,
            "results": [],
        }
        if not survivors:
            return response

        final_candidates, reranker_used = self.retriever.rerank_candidates(query, survivors)
        response["reranker_used"] = reranker_used

        for candidate in final_candidates:
            doc = candidate["doc"]
            chunk_id = make_chunk_id(doc)
            source = describe_source(doc.metadata)
            entry = {
                "chunk_id": chunk_id,
                "title": source["title"],
                "url": source["url"],
                "file_name": source["file_name"],
                "page": source["page"],
                "relevance": round(candidate["relevance"], 3),
                "text": doc.page_content,
                "doc": doc,
            }
            is_new = ledger.add(chunk_id, entry)
            response["results"].append({**entry, "is_new": is_new})
        return response


def search_result_for_model(response: dict, seen_chunk_ids: set) -> dict:
    """Return full text only for chunks not already seen by this model context.

    Each worker has its own context, so the shared ledger cannot determine what that worker has read.
    Chunks already in `seen_chunk_ids` are returned as short references instead of repeating their full text.
    The set is updated in place.
    """
    results = []
    for item in response["results"]:
        if item["chunk_id"] not in seen_chunk_ids:
            seen_chunk_ids.add(item["chunk_id"])
            results.append(
                {
                    "chunk_id": item["chunk_id"],
                    "title": item["title"],
                    "page": item["page"],
                    "relevance": item["relevance"],
                    "text": item["text"],
                }
            )
        else:
            results.append(
                {
                    "chunk_id": item["chunk_id"],
                    "title": item["title"],
                    "page": item["page"],
                    "already_shown": True,
                }
            )
    return {
        "status": response["status"],
        "max_relevance": response["max_relevance"],
        "min_doc_relevance": response["min_doc_relevance"],
        "results": results,
    }


# ================================
# Citation validation
# ================================
def inline_citation_ids(answer: str) -> list:
    """Return the chunk IDs cited inline in the answer text, in order of appearance."""
    ids = []
    for group in INLINE_CITATION_PATTERN.findall(answer):
        for chunk_id in re.split(r"\s*[,;]\s*", group):
            if chunk_id not in ids:
                ids.append(chunk_id)
    return ids


def validate_submission(answer, citations, ledger: EvidenceLedger, max_answer_chars: int) -> list:
    """Validate a submitted answer before acceptance.

    The answer must contain at least one citation, every cited ID must exist in the current request ledger, and inline citations must use `[chunk_id]`.
    The answer must also be non-empty and within the configured length limit.
    These checks prove that citations point to evidence shown to the model.
    Claim-level support is handled by the optional reviewer and measured separately during evaluation.
    The function returns an empty list when the submission is valid.
    """
    errors = []
    listed = [c for c in citations if isinstance(c, str)] if isinstance(citations, list) else []
    inline_ids = inline_citation_ids(answer) if isinstance(answer, str) else []

    # Inline markers and the citation list form one set.
    # Requiring exact agreement caused unnecessary formatting retries.
    if not listed and not inline_ids:
        errors.append("No citations. Cite at least one chunk_id returned by search_documents.")

    # Every ID the model mentions, in the list or in the text, must be in the ledger.
    mentioned = list(listed)
    if isinstance(answer, str):
        mentioned += CHUNK_ID_LIKE_PATTERN.findall(answer)
    unknown = sorted({c for c in mentioned if c not in ledger})
    if unknown:
        errors.append(
            f"Unknown chunk IDs: {', '.join(unknown)}. Only cite chunk_id values "
            "returned by search_documents in this conversation."
        )

    if not isinstance(answer, str) or not answer.strip():
        errors.append("answer is empty.")
        return errors
    if len(answer) > max_answer_chars:
        errors.append(f"answer is too long ({len(answer)} chars, max {max_answer_chars}).")

    if not inline_ids:
        errors.append("Cite evidence inline as [chunk_id] right after the statements it supports.")
    # IDs outside brackets would remain visible instead of being replaced with display numbers.
    stray_ids = CHUNK_ID_LIKE_PATTERN.findall(INLINE_CITATION_PATTERN.sub("", answer))
    if stray_ids:
        errors.append(
            f"Chunk IDs must appear only inside square brackets, e.g. [{stray_ids[0]}]."
        )
    return errors


def number_citations(answer: str, citations: list, ledger: EvidenceLedger):
    """Replace chunk IDs with display numbers and return the cited ledger entries.

    Numbers follow each ID's first appearance in the answer.
    IDs found only in the citation list are numbered afterward.
    """
    order = inline_citation_ids(answer)
    for chunk_id in citations:
        if chunk_id not in order:
            order.append(chunk_id)
    numbers = {chunk_id: index for index, chunk_id in enumerate(order, 1)}

    def replace(match):
        ids = re.split(r"\s*[,;]\s*", match.group(1))
        return "[" + ", ".join(str(numbers[chunk_id]) for chunk_id in ids) + "]"

    display_answer = INLINE_CITATION_PATTERN.sub(replace, answer)
    cited_entries = [ledger.get(chunk_id) for chunk_id in order]
    return display_answer, cited_entries


def validate_findings(findings, citations, ledger: EvidenceLedger) -> list:
    """Validate worker findings and return any errors.

    This is lighter than final-answer validation because findings are notes for the orchestrator.
    Inline markers are optional, but every listed citation must exist in the request ledger.
    """
    errors = []
    if not isinstance(findings, str) or not findings.strip():
        errors.append("findings is empty.")
    listed = [c for c in citations if isinstance(c, str)] if isinstance(citations, list) else []
    if not listed:
        errors.append("citations is empty. Cite the chunk_id values your findings rely on.")
    unknown = sorted({c for c in listed if c not in ledger})
    if unknown:
        errors.append(f"Unknown chunk IDs: {', '.join(unknown)}.")
    return errors


def format_evidence(entries) -> str:
    """Format chunk IDs, titles, pages and full text for another agent.

    Escalated agents, orchestrators and reviewers receive the original ledger text rather than another agent's summary.
    """
    blocks = []
    for entry in entries:
        blocks.append(
            f"[{entry['chunk_id']}] {entry['title']} (p.{entry['page']})\n{entry['text']}"
        )
    return "\n\n---\n\n".join(blocks) if blocks else "(no evidence)"
