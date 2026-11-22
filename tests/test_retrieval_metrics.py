
import pytest
from langchain_core.documents import Document

from evaluation.retrieval_metrics import (
    normalize_document_id,
    get_unique_document_ids,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_hit_rate_at_k,
    evaluate_retrieval_at_k,
)


def test_normalize_document_id_removes_markdown_extension():
    assert normalize_document_id("04_claims_procedure.md") == "04_claims_procedure"


def test_normalize_document_id_handles_windows_path():
    result = normalize_document_id(
        r"C:\documents\04_claims_procedure.md"
    )
    assert result == "04_claims_procedure"


def test_get_unique_document_ids_removes_duplicates():
    documents = [
        Document(
            page_content="Chunk 1",
            metadata={"document_id": "benefits"},
        ),
        Document(
            page_content="Chunk 2",
            metadata={"document_id": "benefits"},
        ),
        Document(
            page_content="Chunk 3",
            metadata={"document_id": "claims"},
        ),
    ]

    assert get_unique_document_ids(documents) == [
        "benefits",
        "claims",
    ]


def test_get_unique_document_ids_uses_source_filename_fallback():
    documents = [
        Document(
            page_content="Claims information",
            metadata={"source_filename": "04_claims_procedure.md"},
        )
    ]

    assert get_unique_document_ids(documents) == [
        "04_claims_procedure"
    ]


def test_calculate_precision_at_k():
    retrieved = ["benefits", "claims", "premium"]
    expected = {"benefits", "claims", "waiting_periods"}

    assert calculate_precision_at_k(retrieved, expected) == pytest.approx(2 / 3)


def test_calculate_recall_at_k():
    retrieved = ["benefits", "claims"]
    expected = {"benefits", "claims", "waiting_periods"}

    assert calculate_recall_at_k(retrieved, expected) == pytest.approx(2 / 3)


def test_calculate_hit_rate_at_k_when_relevant_document_found():
    assert calculate_hit_rate_at_k(
        ["benefits", "claims"],
        {"claims", "waiting_periods"},
    ) == 1


def test_calculate_hit_rate_at_k_when_no_relevant_document_found():
    assert calculate_hit_rate_at_k(
        ["premium", "overview"],
        {"claims", "waiting_periods"},
    ) == 0



def test_evaluate_retrieval_at_k_deduplicates_source_documents():
    documents = [
        Document(page_content="Chunk 1", metadata={"document_id": "benefits"}),
        Document(page_content="Chunk 2", metadata={"document_id": "benefits"}),
        Document(page_content="Chunk 3", metadata={"document_id": "claims"}),
        Document(page_content="Chunk 4", metadata={"document_id": "premium"}),
        Document(page_content="Chunk 5", metadata={"document_id": "waiting_periods"}),
    ]

    result = evaluate_retrieval_at_k(
        documents=documents,
        expected_document_ids=[
            "benefits.md",
            "claims.md",
            "waiting_periods.md",
        ],
        k=3,
    )

    assert result["retrieved_document_ids"] == [
        "benefits",
        "claims",
        "premium",
    ]
    assert result["precision"] == 2 / 3
    assert result["recall"] == 2 / 3
    assert result["hit_rate"] == 1.0



def test_evaluate_retrieval_at_k_rejects_invalid_k():
    with pytest.raises(ValueError):
        evaluate_retrieval_at_k([], ["benefits"], k=0)


def test_evaluate_retrieval_at_k_rejects_empty_expected_sources():
    with pytest.raises(ValueError):
        evaluate_retrieval_at_k([], [], k=3)


def test_metrics_reject_empty_expected_sources():
    with pytest.raises(ValueError):
        calculate_precision_at_k(["benefits"], set())

    with pytest.raises(ValueError):
        calculate_recall_at_k(["benefits"], set())

    with pytest.raises(ValueError):
        calculate_hit_rate_at_k(["benefits"], set())
