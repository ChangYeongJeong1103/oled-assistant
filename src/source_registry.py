"""Map source file names to offline-verified publication titles and DOI links.

`scripts/build_source_registry.py` generates the registry and stores only DOI links confirmed through Crossref.
The model never generates source titles or URLs.
Runtime lookup uses the file name, so attaching this metadata does not require rebuilding ChromaDB.
"""

import json
import os
from functools import lru_cache

import config
from utils import logger


@lru_cache(maxsize=1)
def load_registry(path: str = config.SOURCE_REGISTRY_PATH) -> dict:
    """Load the registry once per process.

    A missing or invalid registry produces a warning and falls back to source file names.
    This preserves the application's behavior before the registry was added.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle).get("sources", {})
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Source registry unavailable (%s). Using file names.", exc)
        return {}


def describe_source(metadata: dict) -> dict:
    """Return display information for one retrieved chunk.

    Metadata contains the source path and may contain a zero-indexed page from `PyPDFLoader`.
    The result contains the file name, display title, optional verified URL and optional one-indexed page.
    Sources without a registry entry use the file name as the title.
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
