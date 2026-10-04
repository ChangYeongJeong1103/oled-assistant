"""
Tools and evidence bookkeeping for the OLED agent.

The agent has no direct access to the vector store. The only search it can do
is call search_documents(query), which runs the same pipeline as the workflow:

    retrieve candidates -> relevance filter -> rerank -> return results

The relevance threshold, the candidate pool size, and the final top-N are all
read from the StrictRAGAssistant instance (i.e. from config). The model only
chooses the query text, so it has no way to lower the threshold or skip the
filter.

Every chunk we return to the model is recorded in an EvidenceLedger that lives
for one request. When the model submits an answer, we check its citations
against that ledger, so it can only cite evidence it was actually shown during
this request.
"""

import hashlib
import os
import re
import threading

from source_registry import describe_source

# Chunk IDs look like "c1a2b3c4d": a "c" followed by the first 8 hex characters
# of a SHA-1 hash. 8 hex characters keep collisions negligible for our ~7k
# chunks, and the ID is still short enough for the model to copy it reliably.
CHUNK_ID_PATTERN = re.compile(r"\bc[0-9a-f]{8}\b")
# Anything that looks like an attempt at a chunk ID, including malformed ones.
CHUNK_ID_LIKE_PATTERN = re.compile(r"\bc[0-9a-f]{5,12}\b")
# One or more IDs inside square brackets, e.g. "[c1a2b3c4d]" or
# "[c1a2b3c4d, c5e6f7a8b]".
INLINE_CITATION_PATTERN = re.compile(r"\[\s*(c[0-9a-f]{8}(?:\s*[,;]\s*c[0-9a-f]{8})*)\s*\]")


class AgentStop(Exception):
    """Raised from anywhere inside a run to end it with the given stop reason."""

    def __init__(self, reason, detail=""):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def make_chunk_id(doc) -> str:
    """
    Build a stable ID for a retrieved chunk.

    Why we need this:
    The results we get from the persisted ChromaDB through LangChain don't
    carry a stable ID, so we hash the things that identify a chunk: its file
    name, page, and text. That way the same chunk gets the same ID in every
    search and every request.
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
    """
    Build the cache key for a search query. Only whitespace is normalized.

    NOTE: We keep signs, decimal points, case, and punctuation as they are,
    because they can change what the query means (e.g. "-5 V" vs "+5 V",
    "1.5 eV" vs "15 eV", "CO" vs "Co").
    """
    return " ".join(query.split())


# ================================
# Evidence ledger
# ================================
class EvidenceLedger:
    """
    Every chunk that any agent was shown during ONE user request.

    We keep insertion order so the sources are always displayed in the same
    order. Workers run in threads and share one ledger, so writes go through
    a lock.
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
    """Wrap the workflow's retrieval methods as the agent's only search tool."""

    def __init__(self, workflow):
        # The StrictRAGAssistant instance owns the vector store, the reranker,
        # and every threshold. We only call its methods here.
        self.workflow = workflow
        # IMPORTANT: Embedding and reranking run on the CPU and share one model
        # instance, so concurrent searches (parallel workers or other users)
        # take turns behind this lock. Running them at the same time would not
        # make them any faster on a 1-CPU pod.
        self._lock = threading.Lock()

    def search(self, query: str, ledger: EvidenceLedger) -> dict:
        """
        Run retrieve, filter, and rerank for one query.

        Args:
            query: The search query chosen by the model.
            ledger: The EvidenceLedger of the current request. Every returned
                chunk is added to it.

        Returns:
            dict with:
              status:        "ok" or "no_relevant_documents"
              max_relevance: the best relevance among ALL candidates, even when
                             none passed the filter, so the agent can judge
                             whether a reworded query might help
              results:       the final chunks (id, title, page, relevance, text, ...)
        """
        with self._lock:
            return self._search(query, ledger)

    def _search(self, query, ledger):
        candidates = self.workflow.retrieve_candidates(query)
        survivors = self.workflow.filter_candidates_by_relevance(candidates)
        max_relevance = max((c["relevance"] for c in candidates), default=0.0)

        response = {
            "status": "ok" if survivors else "no_relevant_documents",
            "query": query,
            "max_relevance": round(max_relevance, 3),
            "min_doc_relevance": self.workflow.min_doc_relevance,
            "candidate_count": len(candidates),
            "survivor_count": len(survivors),
            "reranker_used": False,
            "results": [],
        }
        if not survivors:
            return response

        final_candidates, reranker_used = self.workflow.rerank_candidates(query, survivors)
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
    """
    Shrink a search response down to what the model needs to see.

    seen_chunk_ids holds the chunks that THIS model context has already
    received. Each worker has its own context, so the shared ledger can't
    tell us this. Chunks that are already in the set are sent as a short
    reference instead of repeating their full text. The set is updated in
    place.
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
    """
    Check a submitted answer before we accept it.

    Args:
        answer: The answer text from submit_answer.
        citations: The "citations" list from submit_answer.
        ledger: The EvidenceLedger of the current request.
        max_answer_chars: Longest answer we allow.

    Returns:
        list of error messages. An empty list means the answer is valid.

    Order of checks:
      1) Is there at least one citation (inline [id] or in the citations list)?
      2) Is every cited ID in THIS request's ledger?
      3) Is the answer well-formed (non-empty, within the length limit, cites inline)?

    Note:
        These checks prove that the citations point to evidence the model
        really saw. Whether each claim is actually supported by that evidence
        is checked by the optional reviewer (agent_team.review_answer) and
        measured separately in evaluation.
    """
    errors = []
    listed = [c for c in citations if isinstance(c, str)] if isinstance(citations, list) else []
    inline_ids = inline_citation_ids(answer) if isinstance(answer, str) else []

    # We treat the inline [id] markers and the citations list together as one
    # citation set. Requiring the two to match exactly only caused formatting
    # retries.
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
    # IDs written outside "[...]" would show up in the displayed answer without
    # being replaced by a number.
    stray_ids = CHUNK_ID_LIKE_PATTERN.findall(INLINE_CITATION_PATTERN.sub("", answer))
    if stray_ids:
        errors.append(
            f"Chunk IDs must appear only inside square brackets, e.g. [{stray_ids[0]}]."
        )
    return errors


def number_citations(answer: str, citations: list, ledger: EvidenceLedger):
    """
    Replace the chunk IDs in the answer with [1], [2], ... for display.

    Numbers follow the order in which each ID first appears in the text. IDs
    that only appear in the citations list are numbered after those.

    Returns:
        (display_answer, cited_entries) tuple.
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
    """
    Check a worker's findings and return a list of error messages.

    This is a lighter check than validate_submission. Findings are notes for
    the orchestrator rather than the answer the user reads, so inline markers
    are optional. Every cited ID must still be in this request's ledger.
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
    """
    Render chunks as plain text for a prompt: id, title, page, and full text.

    We use this whenever one agent hands evidence to another (escalation,
    orchestrator, reviewer). The receiving agent reads the ORIGINAL chunk text
    from the ledger instead of another agent's summary of it.
    """
    blocks = []
    for entry in entries:
        blocks.append(
            f"[{entry['chunk_id']}] {entry['title']} (p.{entry['page']})\n{entry['text']}"
        )
    return "\n\n---\n\n".join(blocks) if blocks else "(no evidence)"
