
import pytest
from unittest.mock import Mock
from langchain_core.documents import Document

from retrieval.reranker import rerank_documents


@pytest.fixture
def sample_documents():
    return [
        Document(page_content="General hospitalization eligibility."),
        Document(page_content="Pre-hospitalization: 30 days. Post-hospitalization: 60 days."),
        Document(page_content="Submit claim form, discharge summary, original bills and prescriptions."),
    ]


def test_rerank_documents_by_descending_score(sample_documents):
    mock_model = Mock()
    mock_model.predict.return_value = [0.2, 0.9, 0.7]

    results = rerank_documents(
        query="What are the hospitalization expense windows?",
        documents=sample_documents,
        model=mock_model,
    )

    assert len(results) == 3
    assert results[0][0] == sample_documents[1]
    assert results[1][0] == sample_documents[2]
    assert results[2][0] == sample_documents[0]
    assert results[0][1] == 0.9


def test_rerank_documents_respects_top_k(sample_documents):
    mock_model = Mock()
    mock_model.predict.return_value = [0.2, 0.9, 0.7]

    results = rerank_documents(
        query="What are the hospitalization expense windows?",
        documents=sample_documents,
        model=mock_model,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0][0] == sample_documents[1]


def test_rerank_documents_empty_list():
    mock_model = Mock()

    results = rerank_documents(
        query="Test query",
        documents=[],
        model=mock_model,
    )

    assert results == []
    mock_model.predict.assert_not_called()


def test_rerank_documents_invalid_query():
    with pytest.raises(ValueError):
        rerank_documents(
            query=" ",
            documents=[],
        )


def test_rerank_documents_invalid_documents():
    with pytest.raises(ValueError):
        rerank_documents(
            query="Test query",
            documents="not a list",
        )


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_rerank_documents_invalid_top_k(sample_documents, top_k):
    with pytest.raises(ValueError):
        rerank_documents(
            query="Test query",
            documents=sample_documents,
            top_k=top_k,
        )


def test_rerank_documents_mismatched_scores(sample_documents):
    mock_model = Mock()
    mock_model.predict.return_value = [0.9, 0.5]

    with pytest.raises(ValueError, match="unexpected number of scores"):
        rerank_documents(
            query="Test query",
            documents=sample_documents,
            model=mock_model,
        )



from retrieval.reranker import rerank_documents_for_queries


def test_rerank_documents_for_multiple_queries(sample_documents):
    mock_model = Mock()
    mock_model.predict.side_effect = [
        [0.2, 0.9, 0.1],
        [0.1, 0.2, 0.95],
    ]

    queries = [
        "What are the hospitalization expense windows?",
        "What documents are needed for reimbursement?",
    ]

    results = rerank_documents_for_queries(
        queries=queries,
        documents=sample_documents,
        model=mock_model,
    )

    assert len(results) == 2

    assert results[queries[0]][0][0] == sample_documents[1]
    assert results[queries[1]][0][0] == sample_documents[2]

    assert mock_model.predict.call_count == 2


def test_rerank_documents_for_multiple_queries_respects_top_k(
    sample_documents,
):
    mock_model = Mock()
    mock_model.predict.side_effect = [
        [0.2, 0.9, 0.1],
        [0.1, 0.2, 0.95],
    ]

    results = rerank_documents_for_queries(
        queries=["Query A", "Query B"],
        documents=sample_documents,
        model=mock_model,
        top_k=2,
    )

    assert len(results["Query A"]) == 2
    assert len(results["Query B"]) == 2


def test_rerank_documents_for_multiple_queries_empty_documents():
    mock_model = Mock()

    results = rerank_documents_for_queries(
        queries=["Query A", "Query B"],
        documents=[],
        model=mock_model,
    )

    assert results == {"Query A": [], "Query B": []}
    mock_model.predict.assert_not_called()


def test_rerank_documents_for_multiple_queries_invalid_queries(
    sample_documents,
):
    with pytest.raises(ValueError):
        rerank_documents_for_queries(
            queries=[],
            documents=sample_documents,
        )

    with pytest.raises(ValueError):
        rerank_documents_for_queries(
            queries=["Query A", " "],
            documents=sample_documents,
        )


def test_rerank_documents_for_multiple_queries_duplicate_queries(
    sample_documents,
):
    with pytest.raises(ValueError, match="duplicates"):
        rerank_documents_for_queries(
            queries=["Query A", "Query A"],
            documents=sample_documents,
        )
