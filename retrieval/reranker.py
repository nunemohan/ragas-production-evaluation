
from sentence_transformers import CrossEncoder


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rerank_documents(
    query: str,
    documents: list,
    model=None,
    model_name: str = DEFAULT_RERANKER_MODEL,
    top_k: int | None = None,
) -> list:
    """
    Rerank retrieved documents using a Cross-Encoder.

    Returns:
        List of (document, relevance_score) tuples,
        sorted by descending relevance score.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string")

    if not isinstance(documents, list):
        raise ValueError("Documents must be a list")

    if top_k is not None:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

    if not documents:
        return []

    if model is None:
        model = CrossEncoder(model_name)

    pairs = [
        (query.strip(), document.page_content)
        for document in documents
    ]

    scores = model.predict(pairs)

    if len(scores) != len(documents):
        raise ValueError("Model returned an unexpected number of scores")

    ranked_results = sorted(
        zip(documents, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    if top_k is not None:
        ranked_results = ranked_results[:top_k]

    return [
        (document, float(score))
        for document, score in ranked_results
    ]


def rerank_documents_for_queries(
    queries: list[str],
    documents: list,
    model=None,
    model_name: str = DEFAULT_RERANKER_MODEL,
    top_k: int | None = None,
) -> dict[str, list]:
    """
    Rerank candidate documents independently for multiple queries.

    Returns:
        {
            query_a: [(document, score), ...],
            query_b: [(document, score), ...]
        }
    """

    if not isinstance(queries, list) or not queries:
        raise ValueError("Queries must be a non-empty list")

    for query in queries:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Every query must be a non-empty string")

    if len(set(queries)) != len(queries):
        raise ValueError("Queries must not contain duplicates")

    if not isinstance(documents, list):
        raise ValueError("Documents must be a list")

    if top_k is not None:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

    if not documents:
        return {query: [] for query in queries}

    if model is None:
        model = CrossEncoder(model_name)

    results = {}

    for query in queries:
        results[query] = rerank_documents(
            query=query,
            documents=documents,
            model=model,
            top_k=top_k,
        )

    return results

