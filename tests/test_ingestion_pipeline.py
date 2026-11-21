
import pytest

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ingestion.pipeline import run_ingestion_pipeline


class FakeEmbeddingModel(Embeddings):
    """Fake embeddings for end-to-end pipeline tests."""

    def _embed(self, text: str) -> list[float]:
        text = text.lower()

        if "hospital" in text or "coverage" in text:
            return [1.0, 0.0, 0.0]

        if "claim" in text or "reimbursement" in text:
            return [0.0, 1.0, 0.0]

        return [0.0, 0.0, 1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def create_source_documents(directory):
    directory.mkdir(parents=True, exist_ok=True)

    (directory / "policy_coverage.md").write_text(
        "# Hospitalization Coverage\n\n"
        + ("Hospitalization coverage is available under the policy. " * 30),
        encoding="utf-8",
    )

    (directory / "claims_process.md").write_text(
        "# Claims Process\n\n"
        + ("Claims require bills and reimbursement documents. " * 30),
        encoding="utf-8",
    )


def test_complete_ingestion_pipeline(tmp_path):
    documents_dir = tmp_path / "documents"
    vector_store_dir = tmp_path / "vector_store"

    create_source_documents(documents_dir)

    result = run_ingestion_pipeline(
        documents_dir=str(documents_dir),
        vector_store_dir=str(vector_store_dir),
        chunk_size=100,
        chunk_overlap=20,
        embedding_model=FakeEmbeddingModel(),
    )

    assert result["documents_loaded"] == 2
    assert result["documents_cleaned"] == 2
    assert result["chunks_created"] > 2
    assert result["vectors_stored"] == result["chunks_created"]


def test_pipeline_creates_faiss_files(tmp_path):
    documents_dir = tmp_path / "documents"
    vector_store_dir = tmp_path / "vector_store"

    create_source_documents(documents_dir)

    run_ingestion_pipeline(
        str(documents_dir),
        str(vector_store_dir),
        embedding_model=FakeEmbeddingModel(),
    )

    assert (vector_store_dir / "index.faiss").exists()
    assert (vector_store_dir / "index.pkl").exists()


def test_pipeline_vector_store_can_retrieve_documents(tmp_path):
    documents_dir = tmp_path / "documents"
    vector_store_dir = tmp_path / "vector_store"

    create_source_documents(documents_dir)

    result = run_ingestion_pipeline(
        str(documents_dir),
        str(vector_store_dir),
        embedding_model=FakeEmbeddingModel(),
    )

    retrieved = result["vector_store"].similarity_search(
        "hospitalization coverage",
        k=1,
    )

    assert len(retrieved) == 1
    assert "hospitalization" in retrieved[0].page_content.lower()


def test_missing_documents_directory_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_ingestion_pipeline(
            str(tmp_path / "missing"),
            str(tmp_path / "vector_store"),
            embedding_model=FakeEmbeddingModel(),
        )


def test_empty_documents_directory_raises_error(tmp_path):
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()

    with pytest.raises(ValueError):
        run_ingestion_pipeline(
            str(documents_dir),
            str(tmp_path / "vector_store"),
            embedding_model=FakeEmbeddingModel(),
        )


def test_invalid_chunk_configuration_raises_error(tmp_path):
    documents_dir = tmp_path / "documents"
    create_source_documents(documents_dir)

    with pytest.raises(ValueError):
        run_ingestion_pipeline(
            str(documents_dir),
            str(tmp_path / "vector_store"),
            chunk_size=100,
            chunk_overlap=100,
            embedding_model=FakeEmbeddingModel(),
        )
