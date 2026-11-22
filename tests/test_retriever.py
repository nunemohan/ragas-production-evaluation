
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ingestion.vector_store import build_vector_store
from retrieval.retriever import create_retriever, retrieve_documents


class FakeEmbeddingModel(Embeddings):

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        text = text.lower()

        # Simple deterministic vectors for testing.
        if "hospitalization" in text or "medical" in text:
            return [1.0, 0.0, 0.0]

        if "claim" in text or "reimbursement" in text:
            return [0.0, 1.0, 0.0]

        return [0.0, 0.0, 1.0]


@pytest.fixture
def vector_store(tmp_path):
    documents = [
        Document(
            page_content="Pre hospitalization expenses are covered for 30 days.",
            metadata={"document_id": "benefits"},
        ),
        Document(
            page_content="Post hospitalization expenses are covered for 60 days.",
            metadata={"document_id": "benefits"},
        ),
        Document(
            page_content="Reimbursement claims require original bills and receipts.",
            metadata={"document_id": "claims"},
        ),
    ]

    return build_vector_store(
        documents,
        embedding_model=FakeEmbeddingModel(),
    )


def test_retrieve_documents_returns_results(vector_store):
    results = retrieve_documents(
        vector_store,
        "hospitalization coverage",
        top_k=2,
    )

    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc in results)


def test_retrieve_documents_respects_top_k(vector_store):
    results = retrieve_documents(
        vector_store,
        "hospitalization coverage",
        top_k=1,
    )

    assert len(results) == 1


def test_retrieve_documents_rejects_empty_query(vector_store):
    with pytest.raises(ValueError):
        retrieve_documents(vector_store, "   ")


def test_retrieve_documents_rejects_invalid_top_k(vector_store):
    with pytest.raises(ValueError):
        retrieve_documents(vector_store, "hospitalization", top_k=0)


def test_create_retriever_loads_saved_vector_store(tmp_path):
    documents = [
        Document(
            page_content="Hospitalization benefits are available.",
            metadata={"document_id": "benefits"},
        )
    ]

    fake_model = FakeEmbeddingModel()

    store = build_vector_store(
        documents,
        embedding_model=fake_model,
    )

    store.save_local(str(tmp_path / "faiss_index"))

    loaded_store = create_retriever(
        str(tmp_path / "faiss_index"),
        embedding_model=fake_model,
    )

    results = retrieve_documents(
        loaded_store,
        "hospitalization",
        top_k=1,
    )

    assert len(results) == 1
    assert "Hospitalization" in results[0].page_content


def test_create_retriever_rejects_missing_directory(tmp_path):
    missing_dir = tmp_path / "missing_vector_store"

    with pytest.raises(FileNotFoundError):
        create_retriever(str(missing_dir))


def test_retrieve_documents_with_metadata_filter(vector_store):
    results = retrieve_documents(
        vector_store,
        "hospitalization coverage",
        top_k=3,
        metadata_filter={"document_id": "benefits"},
    )

    assert len(results) > 0
    assert all(
        doc.metadata["document_id"] == "benefits"
        for doc in results
    )


def test_retrieve_documents_returns_only_matching_metadata(vector_store):
    results = retrieve_documents(
        vector_store,
        "hospitalization coverage",
        top_k=3,
        metadata_filter={"document_id": "claims"},
    )

    assert len(results) > 0
    assert all(
        doc.metadata["document_id"] == "claims"
        for doc in results
    )


def test_retrieve_documents_with_no_matching_metadata(vector_store):
    results = retrieve_documents(
        vector_store,
        "hospitalization coverage",
        top_k=3,
        metadata_filter={"document_id": "non_existing"},
    )

    assert results == []


def test_retrieve_documents_rejects_invalid_metadata_filter(vector_store):
    with pytest.raises(ValueError):
        retrieve_documents(
            vector_store,
            "hospitalization coverage",
            metadata_filter="benefits",
        )


def test_retrieve_documents_without_metadata_filter(vector_store):
    results = retrieve_documents(
        vector_store,
        "hospitalization coverage",
        top_k=3,
    )

    assert len(results) == 3

