
import pytest

from langchain_core.documents import Document

from ingestion.embedding_generator import generate_embeddings


class FakeEmbeddingModel:
    """Fake model for unit testing without API calls."""

    def embed_documents(self, texts):
        return [
            [0.1, 0.2, 0.3]
            for _ in texts
        ]


def create_chunks():
    return [
        Document(
            page_content="Policy covers hospitalization.",
            metadata={"document_id": "policy_one"},
        ),
        Document(
            page_content="Claims require supporting documents.",
            metadata={"document_id": "policy_two"},
        ),
    ]


def test_generate_embeddings_for_multiple_chunks():
    vectors = generate_embeddings(
        create_chunks(),
        embedding_model=FakeEmbeddingModel(),
    )

    assert len(vectors) == 2


def test_each_embedding_is_a_numerical_vector():
    vectors = generate_embeddings(
        create_chunks(),
        embedding_model=FakeEmbeddingModel(),
    )

    assert all(
        all(isinstance(value, (int, float)) for value in vector)
        for vector in vectors
    )


def test_embedding_dimension_is_preserved():
    vectors = generate_embeddings(
        create_chunks(),
        embedding_model=FakeEmbeddingModel(),
    )

    assert all(len(vector) == 3 for vector in vectors)


def test_empty_chunk_list_returns_empty_list():
    assert generate_embeddings([]) == []


def test_invalid_input_type_raises_error():
    with pytest.raises(TypeError):
        generate_embeddings(
            "not a list",
            embedding_model=FakeEmbeddingModel(),
        )


def test_invalid_document_type_raises_error():
    with pytest.raises(TypeError):
        generate_embeddings(
            ["not a Document"],
            embedding_model=FakeEmbeddingModel(),
        )


def test_empty_chunk_content_raises_error():
    empty_document = Document(page_content="   ")

    with pytest.raises(ValueError):
        generate_embeddings(
            [empty_document],
            embedding_model=FakeEmbeddingModel(),
        )


def test_embedding_count_mismatch_raises_error():
    class IncorrectEmbeddingModel:
        def embed_documents(self, texts):
            return [[0.1, 0.2, 0.3]]

    with pytest.raises(ValueError):
        generate_embeddings(
            create_chunks(),
            embedding_model=IncorrectEmbeddingModel(),
        )


def test_inconsistent_embedding_dimensions_raise_error():
    class IncorrectEmbeddingModel:
        def embed_documents(self, texts):
            return [
                [0.1, 0.2, 0.3],
                [0.1, 0.2],
            ]

    with pytest.raises(ValueError):
        generate_embeddings(
            create_chunks(),
            embedding_model=IncorrectEmbeddingModel(),
        )


def test_empty_embedding_vector_raises_error():
    class IncorrectEmbeddingModel:
        def embed_documents(self, texts):
            return [[] for _ in texts]

    with pytest.raises(ValueError):
        generate_embeddings(
            create_chunks(),
            embedding_model=IncorrectEmbeddingModel(),
        )
