"""Compare retrieval output across dependency environments with fixed embeddings and a fake reranker.

Run the script with the same persistent database path under the old and new dependency environments, then compare the JSON output.
The script does not load neural weights or contact an LLM.

Usage: `python scripts/check_retrieval_compat.py /tmp/oled-compat-db > result.json`
"""
import hashlib
import json
import math
import pathlib
import sys
import tempfile
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
fake = types.ModuleType("sentence_transformers")


class CrossEncoder:
    """Return deterministic reranker scores without loading a neural model."""

    def __init__(self, *args, **kwargs):
        pass

    def predict(self, pairs):
        return [text.count("OLED") for _, text in pairs]


fake.CrossEncoder = CrossEncoder
sys.modules["sentence_transformers"] = fake

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from document_pipeline import split_documents  # noqa: E402
from retrieval import Retriever  # noqa: E402


class FixtureEmbeddings(Embeddings):
    """Generate stable fixture embeddings from SHA-256 bytes."""

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        values = [value / 255 for value in hashlib.sha256(text.encode()).digest()]
        norm = math.sqrt(sum(value * value for value in values))
        return [value / norm for value in values]


docs = [
    Document(
        page_content="OLED charge transport. " * 220,
        metadata={"source": "fixture.pdf", "page": 0},
    ),
    Document(
        page_content="TADF excitons. " * 280,
        metadata={"source": "fixture.pdf", "page": 1},
    ),
]
chunks = split_documents(docs)
db_path = sys.argv[1]
embeddings = FixtureEmbeddings()
if not pathlib.Path(db_path).exists():
    db = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=db_path,
    )
else:
    db = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
    )

retriever = Retriever(db, 20, 0.5, 2, True, "fixture", 0.6, 17.27)
output = {
    "chunks": [(chunk.page_content, chunk.metadata) for chunk in chunks],
    "queries": [],
}
for query in ["OLED transport", "TADF excitons", "baking a cake"]:
    candidates = retriever.retrieve_candidates(query)
    survivors = retriever.filter_candidates_by_relevance(candidates)
    final, reranker_used = retriever.rerank_candidates(query, survivors)
    output["queries"].append(
        {
            "q": query,
            "candidates": [
                (item["doc"].page_content, round(item["relevance"], 7))
                for item in candidates
            ],
            "survivors": len(survivors),
            "final": [item["doc"].page_content for item in final],
            "reranked": reranker_used,
        }
    )

print(json.dumps(output, sort_keys=True))
