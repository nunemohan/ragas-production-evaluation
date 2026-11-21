
from langchain_core.documents import Document

from ingestion.document_cleaner import (
    clean_document,
    clean_documents,
)


def test_normalize_line_endings():
    document = Document(
        page_content="Policy\r\nCoverage\rClaim",
        metadata={"source": "policy.md"}
    )

    cleaned = clean_document(document)

    assert cleaned.page_content == (
        "Policy\nCoverage\nClaim"
    )


def test_remove_null_characters():
    document = Document(
        page_content="Policy\x00 Coverage"
    )

    cleaned = clean_document(document)

    assert "\x00" not in cleaned.page_content
    assert cleaned.page_content == "Policy Coverage"


def test_normalize_non_breaking_spaces():
    document = Document(
        page_content="Policy\u00a0Coverage"
    )

    cleaned = clean_document(document)

    assert cleaned.page_content == "Policy Coverage"


def test_remove_trailing_whitespace():
    document = Document(
        page_content="Policy   \nCoverage  "
    )

    cleaned = clean_document(document)

    assert cleaned.page_content == "Policy\nCoverage"


def test_collapse_excessive_blank_lines():
    document = Document(
        page_content="Policy\n\n\n\nCoverage"
    )

    cleaned = clean_document(document)

    assert cleaned.page_content == "Policy\n\nCoverage"


def test_preserve_policy_information():
    original_content = (
        "## Insurance Benefits\n\n"
        "Sum insured: ₹5,00,000\n"
        "Room rent: 1%\n"
        "Waiting period: 36 months\n"
    )

    document = Document(
        page_content=original_content,
        metadata={"source": "benefits.md"}
    )

    cleaned = clean_document(document)

    assert "₹5,00,000" in cleaned.page_content
    assert "1%" in cleaned.page_content
    assert "36 months" in cleaned.page_content
    assert "## Insurance Benefits" in cleaned.page_content


def test_preserve_document_metadata():
    metadata = {
        "source": "policy.md",
        "document_id": "policy",
    }

    document = Document(
        page_content="Insurance Policy",
        metadata=metadata
    )

    cleaned = clean_document(document)

    assert cleaned.metadata == metadata
    assert cleaned.metadata is not document.metadata


def test_original_document_is_not_modified():
    document = Document(
        page_content="Policy   \n\n\nCoverage",
        metadata={"source": "policy.md"}
    )

    original_content = document.page_content

    clean_document(document)

    assert document.page_content == original_content


def test_cleaning_is_idempotent():
    document = Document(
        page_content="Policy   \n\n\nCoverage"
    )

    first_clean = clean_document(document)
    second_clean = clean_document(first_clean)

    assert (
        first_clean.page_content
        == second_clean.page_content
    )


def test_clean_multiple_documents():
    documents = [
        Document(page_content="Policy   "),
        Document(page_content="Coverage\n\n\nBenefits"),
    ]

    cleaned = clean_documents(documents)

    assert len(cleaned) == 2
    assert cleaned[0].page_content == "Policy"
    assert cleaned[1].page_content == (
        "Coverage\n\nBenefits"
    )


def test_invalid_input_raises_error():
    import pytest

    with pytest.raises(TypeError):
        clean_document("Not a Document")
