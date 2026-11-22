
from langchain_core.documents import Document

import pytest

from retrieval.rrf import reciprocal_rank_fusion

def make_document(document_id, chunk_index, content):
    return Document(
        page_content=content,
        metadata={
            "document_id": document_id,
            "chunk_index": chunk_index,
        },
    )

def test_rrf_combines_multiple_rankings():
    query_a = [
        make_document("doc_a", 0, "Chunk A"),
        make_document("doc_b", 0, "Chunk B"),
        make_document("doc_c", 0, "Chunk C"),
    ]

    query_b = [
        make_document("doc_c", 0, "Chunk C"),
        make_document("doc_a", 0, "Chunk A"),
        make_document("doc_d", 0, "Chunk D"),
    ]

    results = reciprocal_rank_fusion(
        [query_a, query_b],
        k=60,
    )

    retrieved = [
        document.page_content
        for document, score in results
    ]

    assert retrieved[0] == "Chunk A"
    assert retrieved[1] == "Chunk C"

def test_rrf_removes_duplicate_chunks():
    query_a = [
        make_document("doc_a", 0, "Chunk A"),
        make_document("doc_a", 0, "Chunk A"),
    ]

    results = reciprocal_rank_fusion([query_a])

    assert len(results) == 1

def test_rrf_respects_top_k():
    query_a = [
        make_document("doc_a", 0, "Chunk A"),
        make_document("doc_b", 0, "Chunk B"),
        make_document("doc_c", 0, "Chunk C"),
    ]

    results = reciprocal_rank_fusion(
        [query_a],
        top_k=2,
    )

    assert len(results) == 2

def test_rrf_rejects_invalid_k():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], k=0)

def test_rrf_rejects_invalid_top_k():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], top_k=0)

def test_rrf_rejects_invalid_result_lists():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion("invalid")

def test_rrf_rejects_non_list_query_results():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[make_document("doc_a", 0, "A")], "invalid"])