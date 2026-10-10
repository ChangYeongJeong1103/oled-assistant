"""Safety checks for the persisted ChromaDB lifecycle."""

import pytest

import document_pipeline


def test_open_failure_preserves_existing_database(tmp_path, monkeypatch):
    """A generic open error must never delete the persisted database."""
    database_path = tmp_path / "chroma"
    database_path.mkdir()
    marker = database_path / "keep-me"
    marker.write_text("existing data", encoding="utf-8")

    class BrokenChroma:
        """Simulate a permission, filesystem, version, or corruption error."""

        def __init__(self, **_kwargs):
            raise PermissionError("temporary access failure")

    monkeypatch.setattr(document_pipeline, "Chroma", BrokenChroma)

    with pytest.raises(RuntimeError, match="left unchanged"):
        document_pipeline.get_or_create_vectorstore(
            embeddings=object(),
            docs_folder=str(tmp_path / "source-documents"),
            persist_directory=str(database_path),
        )

    assert marker.read_text(encoding="utf-8") == "existing data"
