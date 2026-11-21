
import pytest

from langchain_core.documents import Document

from ingestion.chunking import split_documents


def create_sample_document(
    content: str,
    document_id: str = "sample_policy",
) -> Document:
    return Document(
        page_content=content,
        metadata={
            "document_id": document_id,
            "source_filename": f"{document_id}.md",
        },
    )


def test_split_long_document_into_multiple_chunks():
    document = create_sample_document(
        "Insurance coverage information. " * 100
    )

    chunks = split_documents([document])

    assert len(chunks) > 1


def test_chunks_do_not_exceed_configured_size():
    document = create_sample_document(
        "Insurance coverage information. " * 100
    )

    chunks = split_documents(
        [document],
        chunk_size=100,
        chunk_overlap=20,
    )

    assert all(len(chunk.page_content) <= 100 for chunk in chunks)


def test_chunk_metadata_is_preserved():
    document = create_sample_document(
        "Insurance coverage information. " * 100
    )

    chunks = split_documents([document])

    for chunk in chunks:
        assert chunk.metadata["document_id"] == "sample_policy"
        assert chunk.metadata["source_filename"] == "sample_policy.md"


def test_chunk_index_is_sequential():
    document = create_sample_document(
        "Insurance coverage information. " * 100
    )

    chunks = split_documents([document])

    indexes = [chunk.metadata["chunk_index"] for chunk in chunks]

    assert indexes == list(range(len(chunks)))


def test_start_index_is_available():
    document = create_sample_document(
        "Insurance coverage information. " * 100
    )

    chunks = split_documents([document])

    assert all("start_index" in chunk.metadata for chunk in chunks)
    assert all(chunk.metadata["start_index"] >= 0 for chunk in chunks)


def test_adjacent_chunks_have_overlap():
    document = create_sample_document(
        "Insurance coverage information. " * 100
    )

    chunks = split_documents(
        [document],
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) > 1

    first = chunks[0]
    second = chunks[1]

    first_end = (
        first.metadata["start_index"] + len(first.page_content)
    )

    assert second.metadata["start_index"] < first_end


def test_multiple_documents_are_split_independently():
    documents = [
        create_sample_document(
            "Policy coverage information. " * 60,
            "policy_one",
        ),
        create_sample_document(
            "Claims procedure information. " * 60,
            "policy_two",
        ),
    ]

    chunks = split_documents(documents)

    document_ids = {
        chunk.metadata["document_id"] for chunk in chunks
    }

    assert document_ids == {"policy_one", "policy_two"}


def test_chunk_index_restarts_for_each_document():
    documents = [
        create_sample_document(
            "Policy coverage information. " * 60,
            "policy_one",
        ),
        create_sample_document(
            "Claims procedure information. " * 60,
            "policy_two",
        ),
    ]

    chunks = split_documents(documents)

    for document_id in ["policy_one", "policy_two"]:
        document_chunks = [
            chunk for chunk in chunks
            if chunk.metadata["document_id"] == document_id
        ]

        indexes = [
            chunk.metadata["chunk_index"]
            for chunk in document_chunks
        ]

        assert indexes == list(range(len(document_chunks)))


def test_empty_document_list_returns_empty_list():
    assert split_documents([]) == []


def test_invalid_document_input_raises_error():
    with pytest.raises(TypeError):
        split_documents("not a list")


def test_invalid_chunk_size_raises_error():
    document = create_sample_document("Policy information.")

    with pytest.raises(ValueError):
        split_documents([document], chunk_size=0)


def test_overlap_must_be_smaller_than_chunk_size():
    document = create_sample_document("Policy information.")

    with pytest.raises(ValueError):
        split_documents(
            [document],
            chunk_size=100,
            chunk_overlap=100,
        )
