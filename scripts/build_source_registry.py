"""
Build src/source_registry.json, which maps every source file name to its paper
title and, when we can verify it, a URL.

Why this script exists
----------------------
Retrieved chunks only carry a file name such as
"[Baldo] Excitonic singlet-triplet ratio in a semiconducting organic thin film.pdf".
In the UI we want to show the published paper title instead and, when we can
prove it, a clickable DOI link. The model is never allowed to invent titles or
URLs, so every link shown to users must come from this registry.

How a link gets "verified"
--------------------------
1. Collect DOI candidates from the PDF metadata stored with each chunk
   (doi, prism:doi, wps-articledoi, or a "doi:10..." string in subject/title).
2. Look up that DOI on Crossref and accept it only if Crossref's title matches
   a title we already have (the PDF metadata title or the file name).
3. If there is no usable DOI, search Crossref by title. We accept a hit only
   when the file-name title is the start of Crossref's title, word for word
   (file names are often truncated), and the hit is a journal or conference
   paper. We skip reprints in books and abstract services, so the link points
   to the original publication.
Anything that fails these checks keeps its title but gets no URL.

The script reads the persisted ChromaDB with the standard-library sqlite3
module in read-only mode, so it needs no embedding model and never changes
the DB.

Usage
-----
    python scripts/build_source_registry.py
"""

import json
import os
import re
import sqlite3
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_FILE = os.path.join(PROJECT_ROOT, "chroma_db", "chroma.sqlite3")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "src", "source_registry.json")

CROSSREF_API = "https://api.crossref.org/works"
USER_AGENT = "oled-assistant-source-registry/1.0"
REQUEST_PAUSE_SECONDS = 0.15  # Be polite to the public Crossref API.

# Metadata keys that may hold a DOI or a title in the PDF metadata.
DOI_KEYS = ("doi", "prism:doi", "wps-articledoi")
TEXT_KEYS = ("title", "dc:title", "subject")
DOI_PATTERN = re.compile(r"10\.\d{4,9}/[^\s;,\"<>]+", re.IGNORECASE)

# Minimum share of our title words that must appear in Crossref's title.
MIN_TITLE_COVERAGE = 0.8
# Title search is riskier than a DOI lookup, so we only run it for titles
# long enough that a word-for-word match cannot be a coincidence.
MIN_SEARCH_WORDS = 6
SEARCH_ROWS = 5
# Title search hits must have one of these types, so we only link original
# publications.
ACCEPTED_SEARCH_TYPES = {"journal-article", "proceedings-article"}
# DOI prefixes of abstract services (ChemInform) that duplicate the original.
SKIPPED_DOI_PREFIXES = ("10.1002/chin.",)
# Supporting-information DOIs look like "<article doi>.s001".
SUPPORTING_INFO_SUFFIX = re.compile(r"\.s\d{3}$")


# ================================
# Reading the vector store
# ================================
def load_source_metadata(db_file):
    """
    Return {file_name: {"titles": [...], "dois": [...]}} from the ChromaDB SQLite file.

    Every chunk repeats its PDF's metadata, so we merge all chunks of a file.
    """
    connection = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
    wanted_keys = ("source",) + DOI_KEYS + TEXT_KEYS
    placeholders = ",".join("?" * len(wanted_keys))
    rows = connection.execute(
        f"SELECT id, key, string_value FROM embedding_metadata WHERE key IN ({placeholders})",
        wanted_keys,
    ).fetchall()
    connection.close()

    # Group the key/value rows by chunk id first.
    chunks = defaultdict(dict)
    for chunk_id, key, value in rows:
        if value:
            chunks[chunk_id][key] = value

    # Then merge chunks into one record per source file.
    sources = defaultdict(lambda: {"titles": [], "dois": []})
    for metadata in chunks.values():
        file_name = os.path.basename(metadata.get("source", ""))
        if not file_name:
            continue
        record = sources[file_name]

        for key in DOI_KEYS:
            if metadata.get(key):
                add_unique(record["dois"], clean_doi(metadata[key]))

        for key in TEXT_KEYS:
            text = metadata.get(key, "")
            # A DOI is sometimes hidden inside the subject or title string.
            for match in DOI_PATTERN.findall(text):
                add_unique(record["dois"], clean_doi(match))
            if key != "subject" and looks_like_title(text):
                add_unique(record["titles"], text.strip())

    return dict(sources)


def add_unique(items, value):
    """Append value to items if it is non-empty and not already present."""
    if value and value not in items:
        items.append(value)


def clean_doi(raw):
    """Normalize a DOI string: drop prefixes and trailing punctuation."""
    doi = raw.strip()
    doi = re.sub(r"^(doi:|https?://(dx\.)?doi\.org/)", "", doi, flags=re.IGNORECASE)
    return doi.rstrip(".);]").lower()


def looks_like_title(text):
    """
    Reject PDF "titles" that are really internal codes.

    Examples we skip: "doi:10.1016/...", "PII: 0022-0248(74)90173-0",
    "c0jm00593b 10735..10746".
    """
    text = text.strip()
    if len(text.split()) < 3:
        return False
    if re.match(r"^(doi:|pii:)", text, flags=re.IGNORECASE):
        return False
    if re.search(r"\d+\.\.\d+", text):
        return False
    return True


# ================================
# Title helpers
# ================================
def title_from_file_name(file_name):
    """
    Turn a file name into a readable fallback title.

    Removes the extension, personal reading tags such as "[Baldo - IMP!]" or
    "!!! 중요 !!!", and list numbering such as "10." at the start.
    """
    title = re.sub(r"\.pdf$|\.docx$", "", file_name, flags=re.IGNORECASE)

    # Keep stripping leading tags ("[..]", "!", "중요", "IMP") and spaces
    # until nothing changes.
    previous = None
    while previous != title:
        previous = title
        title = re.sub(r"^\s*\[[^\]]*\]\s*", "", title)
        title = re.sub(r"^\s*[!]+\s*", "", title)
        title = re.sub(r"^\s*(중요|매우 중요)\s*[!]*\s*", "", title)

    # "10.Electroluminescence ..." -> "Electroluminescence ..."
    title = re.sub(r"^\d+\.(?=\S)", "", title)
    return re.sub(r"\s+", " ", title).strip()


def title_words(text):
    """
    Return the lower-case alphanumeric words we use to compare two titles.

    We normalize the text first so small formatting differences don't break
    a match:
    - NFKC turns ligatures such as "ﬁ" into "fi".
    - HTML tags like "<i>pin</i>" are removed.
    - Every kind of hyphen or dash is dropped, so "Light-Emitting",
      "LightEmitting" and "Light‐Emitting" all become "lightemitting".
    """
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\-\u2010\u2011\u2012\u2013\u2014]", "", text)
    return [word for word in re.findall(r"[a-z0-9]+", text) if len(word) > 1]


def is_word_prefix(file_title, crossref_title):
    """
    Return True when the file-name title is the start of Crossref's title, word for word.

    File names are often cut off, so Crossref's title may have extra words at
    the END. We also ignore a leading label such as "Review paper:". Extra
    words anywhere else mean a different paper (e.g. "Doped organic ..." vs
    "Organic ...").
    """
    file_words = title_words(file_title)
    candidates = [crossref_title]
    if ":" in crossref_title:
        candidates.append(crossref_title.split(":", 1)[1])

    for candidate in candidates:
        crossref_words = title_words(candidate)
        if file_words and crossref_words[: len(file_words)] == file_words:
            return True
    return False


def coverage(reference, candidate):
    """Share of the reference title's words that also appear in the candidate title."""
    reference_words = title_words(reference)
    if not reference_words:
        return 0.0
    candidate_words = set(title_words(candidate))
    hits = sum(1 for word in reference_words if word in candidate_words)
    return hits / len(reference_words)


# ================================
# Crossref lookups
# ================================
def crossref_get(url, attempts=3):
    """
    GET a Crossref URL and return the decoded JSON, or None on failure.

    We retry transient network errors so that one flaky request doesn't
    quietly downgrade a verifiable source to "title only". A 404 means
    Crossref doesn't know the DOI. That is a real answer, so we don't retry it.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            pass
        finally:
            time.sleep(REQUEST_PAUSE_SECONDS)
        # Back off a little longer after each failed attempt.
        time.sleep(1.0 * (attempt + 1))
    return None


def lookup_doi(doi):
    """Return Crossref's work record for a DOI, or None if it does not resolve."""
    payload = crossref_get(f"{CROSSREF_API}/{urllib.parse.quote(doi)}")
    return payload.get("message") if payload else None


def search_title(title):
    """Return Crossref's top title-search hits (possibly empty)."""
    query = urllib.parse.urlencode({"query.bibliographic": title, "rows": SEARCH_ROWS})
    payload = crossref_get(f"{CROSSREF_API}?{query}")
    if not payload:
        return []
    return payload.get("message", {}).get("items", [])


def usable_doi(doi):
    """Map a supporting-information DOI to its article DOI, and return None for abstract services."""
    doi = SUPPORTING_INFO_SUFFIX.sub("", doi.lower())
    if doi.startswith(SKIPPED_DOI_PREFIXES):
        return None
    return doi


def first_title(work):
    """Crossref stores titles as a list; return the first one or an empty string."""
    titles = work.get("title") or []
    return re.sub(r"\s+", " ", titles[0]).strip() if titles else ""


# ================================
# Registry construction
# ================================
def resolve_source(file_name, record):
    """
    Decide the display title and (optional) verified URL for one source file.

    Args:
        file_name: Source file name as stored in the vector store
        record: {"titles": [...], "dois": [...]} from load_source_metadata()

    Returns:
        dict: Registry entry with "title", "url", "doi", and "verified_by"
            ("crossref_doi", "crossref_title_search", or None)
    """
    fallback_title = title_from_file_name(file_name)
    # PDF title metadata helps confirm a DOI, but it is too unreliable to show
    # on its own (e.g. "Microsoft Word - Ch 6 outline.docx"), so unverified
    # sources display the cleaned file name instead.
    reference_titles = record["titles"] + [fallback_title]

    entry = {
        "title": fallback_title,
        "url": None,
        "doi": None,
        "verified_by": None,
    }

    # 1) DOI from the PDF metadata, confirmed by matching titles.
    for raw_doi in record["dois"]:
        doi = usable_doi(raw_doi)
        if not doi:
            continue
        work = lookup_doi(doi)
        if not work:
            continue
        crossref_title = first_title(work)
        best_match = max(coverage(ref, crossref_title) for ref in reference_titles)
        if crossref_title and best_match >= MIN_TITLE_COVERAGE:
            entry.update(
                title=crossref_title,
                url=f"https://doi.org/{doi}",
                doi=doi,
                verified_by="crossref_doi",
            )
            return entry

    # 2) No usable DOI, so fall back to a strict title search. We only do this
    #    for specific (long) titles.
    if len(title_words(fallback_title)) < MIN_SEARCH_WORDS:
        return entry

    matches = {}
    for work in search_title(fallback_title):
        doi = usable_doi(work.get("DOI", ""))
        crossref_title = first_title(work)
        # Skip book reprints, abstracts, and anything that is not the same title.
        if not doi or work.get("type") not in ACCEPTED_SEARCH_TYPES:
            continue
        if is_word_prefix(fallback_title, crossref_title):
            matches[doi] = crossref_title

    # If two different papers share the title, we cannot tell which one the
    # PDF is, so we keep the title and show no link.
    if len(matches) == 1:
        doi, crossref_title = next(iter(matches.items()))
        entry.update(
            title=crossref_title,
            url=f"https://doi.org/{doi}",
            doi=doi,
            verified_by="crossref_title_search",
        )
    return entry


def main():
    """Build the registry for every source in the vector store and save it."""
    sources = load_source_metadata(DB_FILE)
    print(f"Found {len(sources)} source files in the vector store.")

    registry = {}
    for index, (file_name, record) in enumerate(sorted(sources.items()), 1):
        registry[file_name] = resolve_source(file_name, record)
        status = registry[file_name]["verified_by"] or "title only"
        print(f"[{index:3d}/{len(sources)}] {status:22s} {file_name[:70]}")

    verified = sum(1 for entry in registry.values() if entry["url"])
    with open(OUTPUT_FILE, "w", encoding="utf-8") as handle:
        json.dump({"sources": registry}, handle, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"\nSaved {OUTPUT_FILE}")
    print(f"Verified links: {verified}/{len(registry)} (others show the title only)")


if __name__ == "__main__":
    main()
