
from pathlib import Path

import pytest

from ingestion.document_loader import load_markdown_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOCUMENTS_DIR = (
    PROJECT_ROOT / "data" / "documents"
)


def test_load_all_markdown_documents():
    documents = load_markdown_documents(
        str(DOCUMENTS_DIR)
    )

    assert len(documents) == 5


def test_loaded_documents_are_langchain_documents():
    from langchain_core.documents import Document

    documents = load_markdown_documents(
        str(DOCUMENTS_DIR)
    )

    assert all(
        isinstance(document, Document)
        for document in documents
    )


def test_document_metadata_is_preserved():
    documents = load_markdown_documents(
        str(DOCUMENTS_DIR)
    )

    for document in documents:
        assert "source" in document.metadata
        assert "source_filename" in document.metadata
        assert "document_id" in document.metadata


def test_document_content_is_not_empty():
    documents = load_markdown_documents(
        str(DOCUMENTS_DIR)
    )

    for document in documents:
        assert document.page_content.strip()


def test_missing_directory_raises_error(tmp_path):
    missing_dir = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        load_markdown_documents(str(missing_dir))


def test_empty_directory_raises_error(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(ValueError):
        load_markdown_documents(str(empty_dir))
