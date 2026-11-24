
import pytest
from langchain_core.documents import Document

from generation.context_builder import build_context


def test_build_context_with_multiple_documents():
    documents = [
        Document(
            page_content="The sum insured is Rs. 5,00,000.",
            metadata={
                "source_filename": "01_policy_overview.md",
                "chunk_index": 0,
            },
        ),
        Document(
            page_content="Standard room rent is 1% of sum insured.",
            metadata={
                "source_filename": "02_benefits_and_limits.md",
                "chunk_index": 1,
            },
        ),
    ]

    result = build_context(documents)

    assert "[Context 1]" in result
    assert "[Context 2]" in result
    assert "01_policy_overview.md" in result
    assert "02_benefits_and_limits.md" in result
    assert "Rs. 5,00,000" in result
    assert "Standard room rent" in result


def test_build_context_with_empty_list():
    assert build_context([]) == "No relevant context was retrieved."


def test_build_context_skips_empty_content():
    documents = [
        Document(page_content="   ", metadata={}),
        Document(
            page_content="Waiting period is 36 months.",
            metadata={"document_id": "03_waiting_periods"},
        ),
    ]

    result = build_context(documents)

    assert "[Context 1]" in result
    assert "Waiting period is 36 months." in result
    assert "No relevant context" not in result


def test_build_context_with_missing_metadata():
    documents = [
        Document(
            page_content="Emergency notification is required within 48 hours.",
            metadata={},
        )
    ]

    result = build_context(documents)

    assert "Unknown source" in result
    assert "Unknown" in result


def test_build_context_rejects_invalid_input():
    with pytest.raises(ValueError):
        build_context("not a document list")


def test_build_context_rejects_invalid_document():
    with pytest.raises(ValueError):
        build_context(["invalid document"])
