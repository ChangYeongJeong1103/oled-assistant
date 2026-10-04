"""
Source registry lookup: maps a source file name to its paper title and a
verified URL.

The registry file (src/source_registry.json) is generated offline by
scripts/build_source_registry.py, which only stores DOI links it could confirm
through Crossref. Titles and URLs shown to users always come from this
registry, and the model never writes them. Since we look them up by file name
at runtime, attaching this metadata doesn't require a ChromaDB rebuild.
"""

import json
import os
from functools import lru_cache

import config
from utils import logger


@lru_cache(maxsize=1)
def load_registry(path: str = config.SOURCE_REGISTRY_PATH) -> dict:
    """
    Load the registry once per process.

    NOTE: If the file is missing or broken we only log a warning. Sources then
    fall back to their file names, which is how the app behaved before the
    registry existed.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle).get("sources", {})
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Source registry unavailable (%s). Using file names.", exc)
        return {}


def describe_source(metadata: dict) -> dict:
    """
    Build the display information for one retrieved chunk.

    Args:
        metadata: The chunk's metadata, with "source" (file path) and an
            optional 0-indexed "page" set by PyPDFLoader.

    Returns:
        dict with file_name, title, url (or None), and a 1-indexed page (or None).
        When the file has no registry entry, the title is just the file name.
    """
    file_name = os.path.basename(metadata.get("source", "") or "") or "Unknown source"
    entry = load_registry().get(file_name, {})

    page = metadata.get("page")
    return {
        "file_name": file_name,
        "title": entry.get("title") or file_name,
        "url": entry.get("url"),
        "page": page + 1 if isinstance(page, int) else None,
    }
