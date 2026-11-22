
import json

import pytest
from langchain_core.documents import Document

import evaluation.evaluator as evaluator


@pytest.fixture
def sample_dataset(tmp_path):
    dataset = {
        "dataset_name": "Test Golden Dataset",
        "count": 4,
        "records": [
            {
                "id": "Q01",
                "category": "simple_factual",
                "user_input": "Question about benefits",
                "source_document_ids": [
                    "01_policy_overview.md",
                    "02_benefits_and_limits.md",
                ],
            },
            {
                "id": "Q02",
                "category": "simple_factual",
                "user_input": "Question about claims",
                "source_document_ids": [
                    "04_claims_procedure.md",
                ],
            },
            {
                "id": "Q03",
                "category": "unanswerable",
                "user_input": "Question about unavailable information",
                "source_document_ids": [],
            },
            {
                "id": "Q04",
                "category": "unanswerable",
                "user_input": "Another unavailable question",
                "source_document_ids": [],
            },
        ],
    }

    path = tmp_path / "golden_dataset.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")

    return path


def test_evaluator_processes_all_records(
    sample_dataset, monkeypatch
):
    def fake_retrieve_documents(vector_store, query, top_k):
        if query == "Question about benefits":
            return [
                Document(
                    page_content="Policy overview",
                    metadata={"document_id": "01_policy_overview"},
                ),
                Document(
                    page_content="Claims information",
                    metadata={"document_id": "04_claims_procedure"},
                ),
                Document(
                    page_content="Benefits information",
                    metadata={"document_id": "02_benefits_and_limits"},
                ),
            ]

        if query == "Question about claims":
            return [
                Document(
                    page_content="Claims information",
                    metadata={"document_id": "04_claims_procedure"},
                )
            ]

        if query == "Question about unavailable information":
            return [
                Document(
                    page_content="Premium information",
                    metadata={"document_id": "05_premium_and_servicing"},
                )
            ]

        return []

    monkeypatch.setattr(
        evaluator,
        "retrieve_documents",
        fake_retrieve_documents,
    )

    result = evaluator.evaluate_golden_dataset(
        dataset_path=str(sample_dataset),
        vector_store=object(),
        k_values=(1, 3, 5),
    )

    assert result["total_questions"] == 4
    assert result["answerable_questions"] == 2
    assert result["unanswerable_questions"] == 2
    assert len(result["question_results"]) == 4


def test_evaluator_calculates_metrics_correctly(
    sample_dataset, monkeypatch
):
    def fake_retrieve_documents(vector_store, query, top_k):
        return [
            Document(
                page_content="Overview",
                metadata={"document_id": "01_policy_overview"},
            ),
            Document(
                page_content="Claims",
                metadata={"document_id": "04_claims_procedure"},
            ),
            Document(
                page_content="Benefits",
                metadata={"document_id": "02_benefits_and_limits"},
            ),
        ]

    monkeypatch.setattr(
        evaluator,
        "retrieve_documents",
        fake_retrieve_documents,
    )

    result = evaluator.evaluate_golden_dataset(
        str(sample_dataset),
        object(),
        k_values=(1, 3),
    )

    q1 = result["question_results"][0]

    assert q1["metrics"]["@1"]["precision"] == pytest.approx(1.0)
    assert q1["metrics"]["@1"]["recall"] == pytest.approx(0.5)
    assert q1["metrics"]["@1"]["hit_rate"] == 1

    assert q1["metrics"]["@3"]["precision"] == pytest.approx(2 / 3)
    assert q1["metrics"]["@3"]["recall"] == pytest.approx(1.0)
    assert q1["metrics"]["@3"]["hit_rate"] == 1


def test_evaluator_excludes_unanswerable_from_aggregate_metrics(
    sample_dataset, monkeypatch
):
    def fake_retrieve_documents(vector_store, query, top_k):
        if "benefits" in query:
            return [
                Document(
                    page_content="Benefits",
                    metadata={"document_id": "02_benefits_and_limits"},
                )
            ]

        return [
            Document(
                page_content="Unrelated premium information",
                metadata={"document_id": "05_premium_and_servicing"},
            )
        ]

    monkeypatch.setattr(
        evaluator,
        "retrieve_documents",
        fake_retrieve_documents,
    )

    result = evaluator.evaluate_golden_dataset(
        str(sample_dataset),
        object(),
        k_values=(1,),
    )

    q3 = result["question_results"][2]

    assert q3["is_unanswerable"] is True
    assert q3["metrics"]["@1"]["precision"] is None
    assert q3["metrics"]["@1"]["recall"] is None
    assert q3["metrics"]["@1"]["hit_rate"] is None
    assert q3["metrics"]["@1"]["retriever_returned_chunks"] is True


def test_evaluator_rejects_invalid_k_values(sample_dataset):
    with pytest.raises(ValueError):
        evaluator.evaluate_golden_dataset(
            str(sample_dataset),
            object(),
            k_values=(0, 3),
        )


def test_evaluator_rejects_dataset_count_mismatch(tmp_path):
    dataset = {
        "dataset_name": "Invalid Dataset",
        "count": 2,
        "records": [
            {
                "id": "Q01",
                "user_input": "Test question",
                "source_document_ids": ["01_policy_overview.md"],
            }
        ],
    }

    path = tmp_path / "invalid_dataset.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")

    with pytest.raises(ValueError, match="count"):
        evaluator.evaluate_golden_dataset(
            str(path),
            object(),
        )
