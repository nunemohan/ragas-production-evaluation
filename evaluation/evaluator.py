
import json
from pathlib import Path

from evaluation.retrieval_metrics import (
    evaluate_retrieval_at_k,
    get_unique_document_ids,
)
from retrieval.retriever import retrieve_documents


def evaluate_golden_dataset(
    dataset_path: str,
    vector_store,
    k_values=(1, 3, 5),
):
    """
    Evaluate FAISS retrieval against the golden dataset.

    Metrics are calculated at source-document level.
    Unanswerable questions are reported separately.
    """

    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Golden dataset not found: {dataset_path}"
        )

    if not isinstance(k_values, (list, tuple)) or not k_values:
        raise ValueError("k_values must be a non-empty list or tuple")

    if any(
        not isinstance(k, int) or isinstance(k, bool) or k <= 0
        for k in k_values
    ):
        raise ValueError("All k_values must be positive integers")

    if len(set(k_values)) != len(k_values):
        raise ValueError("k_values must not contain duplicates")

    k_values = tuple(sorted(k_values))

    dataset = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(dataset, dict) or not isinstance(
        dataset.get("records"), list
    ):
        raise ValueError("Invalid golden dataset structure")

    records = dataset["records"]

    if dataset.get("count") != len(records):
        raise ValueError("Dataset count does not match records length")

    question_results = []

    for record in records:
        question_id = record["id"]
        query = record["user_input"]
        expected_sources = record.get("source_document_ids", [])

        # Retrieve the maximum required K only once per question.
        candidate_k = getattr(
        getattr(vector_store, "index", None),
        "ntotal",
        max(k_values),
        )

        retrieved_documents = retrieve_documents(
            vector_store=vector_store,
            query=query,
            top_k=candidate_k,
        )

        is_unanswerable = not expected_sources

        result = {
            "question_id": question_id,
            "category": record.get("category"),
            "user_input": query,
            "is_unanswerable": is_unanswerable,
            "metrics": {},
        }

        for k in k_values:
            if is_unanswerable:
                retrieved_ids = get_unique_document_ids(
                    retrieved_documents[:k]
                )

                result["metrics"][f"@{k}"] = {
                    "precision": None,
                    "recall": None,
                    "hit_rate": None,
                    "retrieved_document_ids": retrieved_ids,
                    "retriever_returned_chunks": bool(
                        retrieved_documents[:k]
                    ),
                }
            else:
                result["metrics"][f"@{k}"] = (
                    evaluate_retrieval_at_k(
                        documents=retrieved_documents,
                        expected_document_ids=expected_sources,
                        k=k,
                    )
                )

        question_results.append(result)

    # Aggregate answerable questions only.
    answerable_results = [
        result
        for result in question_results
        if not result["is_unanswerable"]
    ]

    summary = {}

    for k in k_values:
        metric_key = f"@{k}"

        summary[metric_key] = {}

        for metric_name in ("precision", "recall", "hit_rate"):
            values = [
                result["metrics"][metric_key][metric_name]
                for result in answerable_results
            ]

            summary[metric_key][metric_name] = (
                sum(values) / len(values) if values else None
            )

    return {
        "dataset_name": dataset.get("dataset_name"),
        "total_questions": len(records),
        "answerable_questions": len(answerable_results),
        "unanswerable_questions": (
            len(records) - len(answerable_results)
        ),
        "summary": summary,
        "question_results": question_results,
    }
