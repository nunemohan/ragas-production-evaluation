
import pytest

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


from ingestion.vector_store import (
    build_vector_store,
    save_vector_store,
    load_vector_store,
    search_similar_documents,
)


class FakeEmbeddingModel(Embeddings):
    """Deterministic embeddings for FAISS unit tests."""

    def _embed(self, text: str) -> list[float]:
        text = text.lower()

        if "hospital" in text or "coverage" in text:
            return [1.0, 0.0, 0.0]

        if "claim" in text or "reimbursement" in text:
            return [0.0, 1.0, 0.0]

        return [0.0, 0.0, 1.0]

    def embed_documents(
        self, texts: list[str]
    ) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def create_documents() -> list[Document]:
    return [
        Document(
            page_content="Hospitalization coverage is available under the policy.",
            metadata={
                "document_id": "policy_one",
                "chunk_index": 0,
            },
        ),
        Document(
            page_content="Claims require bills and discharge summaries.",
            metadata={
                "document_id": "policy_two",
                "chunk_index": 0,
            },
        ),
        Document(
            page_content="Reimbursement claims require supporting documents.",
            metadata={
                "document_id": "policy_two",
                "chunk_index": 1,
            },
        ),
    ]


def test_build_faiss_vector_store():
    store = build_vector_store(
        create_documents(),
        embedding_model=FakeEmbeddingModel(),
    )

    assert store.index.ntotal == 3


def test_retrieve_relevant_hospitalization_document():
    store = build_vector_store(
        create_documents(),
        embedding_model=FakeEmbeddingModel(),
    )

    results = search_similar_documents(
        store,
        query="hospitalization coverage",
        top_k=1,
    )

    assert len(results) == 1
    assert "Hospitalization coverage" in results[0].page_content


def test_retrieve_relevant_claims_document():
    store = build_vector_store(
        create_documents(),
        embedding_model=FakeEmbeddingModel(),
    )

    results = search_similar_documents(
        store,
        query="claims reimbursement",
        top_k=1,
    )

    assert len(results) == 1
    assert "Claims require bills" in results[0].page_content


def test_top_k_controls_number_of_results():
    store = build_vector_store(
        create_documents(),
        embedding_model=FakeEmbeddingModel(),
    )

    results = search_similar_documents(
        store,
        query="hospitalization coverage",
        top_k=2,
    )

    assert len(results) == 2


def test_retrieved_metadata_is_preserved():
    store = build_vector_store(
        create_documents(),
        embedding_model=FakeEmbeddingModel(),
    )

    results = search_similar_documents(
        store,
        query="hospitalization coverage",
        top_k=1,
    )

    assert results[0].metadata["document_id"] == "policy_one"
    assert results[0].metadata["chunk_index"] == 0


def test_save_and_load_vector_store(tmp_path):
    documents = create_documents()
    embedding_model = FakeEmbeddingModel()

    store = build_vector_store(
        documents,
        embedding_model=embedding_model,
    )

    save_vector_store(store, str(tmp_path))

    loaded_store = load_vector_store(
        str(tmp_path),
        embedding_model=embedding_model,
    )

    assert loaded_store.index.ntotal == 3


def test_search_after_loading_vector_store(tmp_path):
    documents = create_documents()
    embedding_model = FakeEmbeddingModel()

    store = build_vector_store(
        documents,
        embedding_model=embedding_model,
    )

    save_vector_store(store, str(tmp_path))

    loaded_store = load_vector_store(
        str(tmp_path),
        embedding_model=embedding_model,
    )

    results = search_similar_documents(
        loaded_store,
        query="hospitalization coverage",
        top_k=1,
    )

    assert len(results) == 1
    assert "Hospitalization coverage" in results[0].page_content


def test_empty_document_list_raises_error():
    with pytest.raises(ValueError):
        build_vector_store(
            [],
            embedding_model=FakeEmbeddingModel(),
        )


def test_invalid_document_input_raises_error():
    with pytest.raises(TypeError):
        build_vector_store(
            "not a list",
            embedding_model=FakeEmbeddingModel(),
        )


def test_invalid_document_item_raises_error():
    with pytest.raises(TypeError):
        build_vector_store(
            ["not a Document"],
            embedding_model=FakeEmbeddingModel(),
        )


def test_empty_chunk_content_raises_error():
    with pytest.raises(ValueError):
        build_vector_store(
            [Document(page_content="   ")],
            embedding_model=FakeEmbeddingModel(),
        )


def test_empty_query_raises_error():
    store = build_vector_store(
        create_documents(),
        embedding_model=FakeEmbeddingModel(),
    )

    with pytest.raises(ValueError):
        search_similar_documents(store, query="   ")


def test_invalid_top_k_raises_error():
    store = build_vector_store(
        create_documents(),
        embedding_model=FakeEmbeddingModel(),
    )

    with pytest.raises(ValueError):
        search_similar_documents(
            store,
            query="hospitalization",
            top_k=0,
        )


def test_missing_vector_store_directory_raises_error(tmp_path):
    missing_path = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        load_vector_store(
            str(missing_path),
            embedding_model=FakeEmbeddingModel(),
        )


def test_invalid_vector_store_type_raises_error():
    with pytest.raises(TypeError):
        save_vector_store("not a FAISS store", "unused")
