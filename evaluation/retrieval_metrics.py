
def normalize_document_id(document_id: str) -> str:
    """Normalize document IDs for consistent comparison."""
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be a non-empty string")

    normalized = document_id.strip().replace("\\", "/")
    normalized = normalized.split("/")[-1]

    if normalized.lower().endswith(".md"):
        normalized = normalized[:-3]

    return normalized

def get_unique_document_ids(documents):
    """Extract unique document IDs, preserving retrieval order."""
    unique_ids = []

    for document in documents:
        document_id = document.metadata.get("document_id")

        if not document_id:
            source_filename = document.metadata.get("source_filename")

            if not source_filename:
                raise ValueError(
                    "Retrieved document has no document identifier"
                )

            document_id = source_filename

        normalized_id = normalize_document_id(document_id)

        if normalized_id not in unique_ids:
            unique_ids.append(normalized_id)

    return unique_ids

def calculate_precision_at_k(retrieved_ids, expected_ids):
    if not expected_ids:
        raise ValueError(
            "Precision is undefined for questions without expected sources"
        )

    if not retrieved_ids:
        return 0.0

    relevant_count = len(set(retrieved_ids) & set(expected_ids))

    return relevant_count / len(retrieved_ids)

def calculate_recall_at_k(retrieved_ids, expected_ids):
    if not expected_ids:
        raise ValueError(
            "Recall is undefined for questions without expected sources"
        )

    relevant_count = len(set(retrieved_ids) & set(expected_ids))

    return relevant_count / len(expected_ids)

def calculate_hit_rate_at_k(retrieved_ids, expected_ids):
    if not expected_ids:
        raise ValueError(
            "Hit Rate is undefined for questions without expected sources"
        )

    return int(bool(set(retrieved_ids) & set(expected_ids)))

def evaluate_retrieval_at_k(documents, expected_document_ids, k):
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")

    if not expected_document_ids:
        raise ValueError(
            "Expected document IDs cannot be empty"
        )

    expected_ids = {
        normalize_document_id(doc_id)
        for doc_id in expected_document_ids
    }

    # Limit chunks first, then deduplicate source documents.
    retrieved_ids = get_unique_document_ids(documents)[:k]

    return {
        "k": k,
        "retrieved_document_ids": retrieved_ids,
        "expected_document_ids": sorted(expected_ids),
        "precision": calculate_precision_at_k(
            retrieved_ids, expected_ids
        ),
        "recall": calculate_recall_at_k(
            retrieved_ids, expected_ids
        ),
        "hit_rate": calculate_hit_rate_at_k(
            retrieved_ids, expected_ids
        ),
    }