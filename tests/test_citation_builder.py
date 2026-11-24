
import pytest

from langchain_core.documents import Document

from generation.citation_builder import build_citations


def test_build_citation_from_complete_metadata():
    documents = [
        Document(
            page_content="Policy benefits",
            metadata={
                "document_id": "02_benefits_and_limits",
                "source_filename": "02_benefits_and_limits.md",
                "chunk_index": 2,
            },
        )
    ]

    citations = build_citations(documents)

    assert citations == [
        {
            "document_id": "02_benefits_and_limits",
            "source": "02_benefits_and_limits.md",
            "chunk_index": 2,
        }
    ]


def test_build_citation_using_source_fallback():
    documents = [
        Document(
            page_content="Policy benefits",
            metadata={
                "source": "data/documents/02_benefits_and_limits.md",
                "chunk_index": 1,
            },
        )
    ]

    citations = build_citations(documents)

    assert citations[0]["source"] == "02_benefits_and_limits.md"
    assert citations[0]["document_id"] == "02_benefits_and_limits"


def test_missing_chunk_index():
    documents = [
        Document(
            page_content="Policy benefits",
            metadata={
                "document_id": "benefits",
                "source_filename": "benefits.md",
            },
        )
    ]

    citations = build_citations(documents)

    assert citations == [
        {
            "document_id": "benefits",
            "source": "benefits.md",
        }
    ]


def test_duplicate_citations_are_removed():
    document = Document(
        page_content="Policy benefits",
        metadata={
            "document_id": "benefits",
            "source_filename": "benefits.md",
            "chunk_index": 1,
        },
    )

    citations = build_citations([document, document])

    assert len(citations) == 1


def test_different_chunks_are_preserved():
    documents = [
        Document(
            page_content="Chunk one",
            metadata={
                "document_id": "benefits",
                "source_filename": "benefits.md",
                "chunk_index": 1,
            },
        ),
        Document(
            page_content="Chunk two",
            metadata={
                "document_id": "benefits",
                "source_filename": "benefits.md",
                "chunk_index": 2,
            },
        ),
    ]

    citations = build_citations(documents)

    assert len(citations) == 2


def test_documents_without_source_metadata_are_skipped():
    documents = [
        Document(
            page_content="Some content",
            metadata={},
        )
    ]

    assert build_citations(documents) == []


def test_empty_document_list():
    assert build_citations([]) == []


def test_invalid_input_type():
    with pytest.raises(ValueError):
        build_citations("invalid")


def test_invalid_document_item():
    with pytest.raises(ValueError):
        build_citations(["invalid"])
