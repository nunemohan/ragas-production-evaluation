
from collections import defaultdict


def reciprocal_rank_fusion(
    result_lists,
    k=60,
    top_k=None,
):
    """
    Combine multiple ranked document lists using
    Reciprocal Rank Fusion (RRF).

    Args:
        result_lists: List of ranked Document lists.
        k: RRF ranking constant. Default is 60.
        top_k: Maximum number of fused results.

    Returns:
        List of (Document, RRF score) tuples.
    """

    if not isinstance(result_lists, list):
        raise ValueError("result_lists must be a list")

    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")

    if top_k is not None:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

    scores = defaultdict(float)
    documents = {}
    first_seen = {}

    for result_list in result_lists:
        if not isinstance(result_list, list):
            raise ValueError("Each result set must be a list")

        seen_in_query = set()

        for rank, document in enumerate(result_list, start=1):
            metadata = document.metadata

            document_id = metadata.get("document_id")
            chunk_index = metadata.get("chunk_index")

            if document_id is not None and chunk_index is not None:
                chunk_id = (document_id, chunk_index)
            else:
                chunk_id = document.page_content

            if chunk_id in seen_in_query:
                continue

            seen_in_query.add(chunk_id)

            if chunk_id not in documents:
                documents[chunk_id] = document
                first_seen[chunk_id] = len(first_seen)

            scores[chunk_id] += 1 / (k + rank)

    ranked_chunks = sorted(
        scores,
        key=lambda chunk_id: (
            -scores[chunk_id],
            first_seen[chunk_id],
        ),
    )

    if top_k is not None:
        ranked_chunks = ranked_chunks[:top_k]

    return [
        (documents[chunk_id], scores[chunk_id])
        for chunk_id in ranked_chunks
    ]
