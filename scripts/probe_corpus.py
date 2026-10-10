"""Print the chunks returned by the production retrieval pipeline.

This script is used to ground new evaluation questions.
A question is added to `eval/questions.json` only after its expected points are confirmed in retrieved chunks.

Usage:
`python scripts/probe_corpus.py "what is an exciton" "TADF roll-off"`
`python scripts/probe_corpus.py --chars 900 "exciplex co-host red phosphorescent"`
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from agent_tools import EvidenceLedger, SearchTool  # noqa: E402
from retrieval import build_retriever  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("queries", nargs="+")
    parser.add_argument("--chars", type=int, default=500, help="Characters of chunk text to print")
    args = parser.parse_args()

    # Use the production search tool so the probe matches agent retrieval.
    search_tool = SearchTool(build_retriever())

    for query in args.queries:
        response = search_tool.search(query, EvidenceLedger())
        print(f"\n######## {query}")
        print(f"status={response['status']} max_relevance={response['max_relevance']:.3f}")
        for item in response["results"]:
            text = " ".join(item["text"].split())[: args.chars]
            print(f"\n  [{item['relevance']:.3f}] {item['file_name']} p.{item['page']}\n  {text}")


if __name__ == "__main__":
    main()
