"""Load source documents and manage the OLED Assistant vector store.

The module reuses an existing ChromaDB when available.
It builds a new vector database from source documents only when the database is missing.
"""

import glob
import os
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import config
from utils import logger


def load_documents(docs_folder: str = config.DOCS_FOLDER):
    """Load all PDF and DOCX documents from the data folder."""
    if not os.path.exists(docs_folder):
        raise FileNotFoundError(
            f"Documents folder does not exist: {docs_folder}. "
            "Create this folder and add PDF or DOCX files."
        )

    pdf_files = glob.glob(os.path.join(docs_folder, "*.pdf"))
    docx_files = glob.glob(os.path.join(docs_folder, "*.docx"))

    if not pdf_files and not docx_files:
        raise ValueError(
            f"No PDF or DOCX files found in {docs_folder}. "
            "Add source documents to build the vector store."
        )

    all_documents = []
    for pdf_file in pdf_files:
        logger.info("Loading PDF: %s", os.path.basename(pdf_file))
        all_documents.extend(PyPDFLoader(pdf_file).load())

    for docx_file in docx_files:
        logger.info("Loading DOCX: %s", os.path.basename(docx_file))
        all_documents.extend(Docx2txtLoader(docx_file).load())

    if not all_documents:
        raise ValueError("Document loading failed. No readable content was found.")

    logger.info("Loaded %d document sections.", len(all_documents))
    return all_documents


def split_documents(documents) -> List:
    """Split raw documents into retrieval chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    logger.info(
        "Created %d chunks (chunk_size=%d, chunk_overlap=%d).",
        len(chunks),
        config.CHUNK_SIZE,
        config.CHUNK_OVERLAP,
    )
    return chunks


def _detect_device() -> str:
    """Detect best available device for embeddings."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        # Fall back to CPU when torch probing fails.
        pass

    return "cpu"


def create_embeddings_model() -> HuggingFaceEmbeddings:
    """Create the embedding model used for both indexing and retrieval."""
    device = _detect_device()
    logger.info("Using device for embeddings: %s", device)

    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": config.EMBEDDING_BATCH_SIZE,
        },
    )


def create_vectorstore_with_chroma(
    docs,
    embeddings: Optional[HuggingFaceEmbeddings] = None,
    persist_directory: str = config.DB_PATH,
) -> Chroma:
    """Create and persist ChromaDB from chunked documents."""
    if embeddings is None:
        embeddings = create_embeddings_model()

    logger.info(
        "Creating new ChromaDB at %s from %d chunks.",
        persist_directory,
        len(docs),
    )
    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory,
    )


def build_vectorstore_pipeline(
    docs_folder: str = config.DOCS_FOLDER,
    persist_directory: str = config.DB_PATH,
    embeddings: Optional[HuggingFaceEmbeddings] = None,
) -> Chroma:
    """Load -> split -> index documents into ChromaDB."""
    raw_docs = load_documents(docs_folder)
    chunked_docs = split_documents(raw_docs)
    return create_vectorstore_with_chroma(
        docs=chunked_docs,
        embeddings=embeddings,
        persist_directory=persist_directory,
    )


def get_or_create_vectorstore(
    embeddings: Optional[HuggingFaceEmbeddings] = None,
    docs_folder: str = config.DOCS_FOLDER,
    persist_directory: str = config.DB_PATH,
) -> Chroma:
    """Reuse an existing ChromaDB or build a new one when none exists."""
    if embeddings is None:
        embeddings = create_embeddings_model()

    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        logger.info("Found existing ChromaDB at %s. Trying to reuse it.", persist_directory)
        try:
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
            )
            # Force a lightweight DB call to detect schema/version mismatch early.
            _ = vectorstore._collection.count()
            logger.info("Existing ChromaDB is compatible. Reusing persisted DB.")
            return vectorstore
        except Exception as exc:  # noqa: BLE001
            # Never delete a persisted database after a generic open failure.
            # Permission errors, temporary filesystem problems and corruption require different recovery steps.
            # Rebuilding must therefore be an explicit operation.
            logger.error(
                "Could not open the existing ChromaDB at %s: %s",
                persist_directory,
                exc,
            )
            raise RuntimeError(
                "The existing ChromaDB could not be opened and was left unchanged. "
                "Back it up and rebuild it explicitly from the source documents."
            ) from exc

    logger.info(
        "ChromaDB not found at %s. Creating from documents in %s.",
        persist_directory,
        docs_folder,
    )
    return build_vectorstore_pipeline(
        docs_folder=docs_folder,
        persist_directory=persist_directory,
        embeddings=embeddings,
    )
