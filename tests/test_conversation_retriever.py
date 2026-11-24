
import pytest
from unittest.mock import Mock

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

from retrieval.conversation_retriever import retrieve_with_conversation


@pytest.fixture
def mock_vector_store():
    return Mock()


@pytest.fixture
def mock_documents():
    return [
        Document(
            page_content="Post-hospitalization expenses are covered for 60 days.",
            metadata={"document_id": "benefits"},
        )
    ]


def test_retrieve_with_conversation(
    monkeypatch, mock_vector_store, mock_documents
):
    def fake_rewrite_question(question, chat_history, llm):
        return "What are the post-hospitalization benefits?"

    def fake_retrieve_documents(
        vector_store, query, top_k, metadata_filter
    ):
        assert query == "What are the post-hospitalization benefits?"
        return mock_documents

    monkeypatch.setattr(
        "retrieval.conversation_retriever.rewrite_question",
        fake_rewrite_question,
    )
    monkeypatch.setattr(
        "retrieval.conversation_retriever.retrieve_documents",
        fake_retrieve_documents,
    )

    history = [
        HumanMessage(content="What are the hospitalization benefits?"),
        AIMessage(content="Pre and post-hospitalization are covered."),
    ]

    result = retrieve_with_conversation(
        vector_store=mock_vector_store,
        question="What about after discharge?",
        chat_history=history,
    )

    assert result["original_question"] == "What about after discharge?"
    assert result["rewritten_question"] == (
        "What are the post-hospitalization benefits?"
    )
    assert result["documents"] == mock_documents


def test_retrieval_without_history(
    monkeypatch, mock_vector_store, mock_documents
):
    def fake_rewrite_question(question, chat_history, llm):
        assert chat_history is None
        return question

    def fake_retrieve_documents(
        vector_store, query, top_k, metadata_filter
    ):
        assert query == "What is the policy coverage?"
        return mock_documents

    monkeypatch.setattr(
        "retrieval.conversation_retriever.rewrite_question",
        fake_rewrite_question,
    )
    monkeypatch.setattr(
        "retrieval.conversation_retriever.retrieve_documents",
        fake_retrieve_documents,
    )

    result = retrieve_with_conversation(
        vector_store=mock_vector_store,
        question="What is the policy coverage?",
    )

    assert result["rewritten_question"] == "What is the policy coverage?"
    assert result["documents"] == mock_documents


def test_retrieval_passes_top_k_and_filter(
    monkeypatch, mock_vector_store
):
    captured = {}

    def fake_rewrite_question(question, chat_history, llm):
        return question

    def fake_retrieve_documents(
        vector_store, query, top_k, metadata_filter
    ):
        captured["top_k"] = top_k
        captured["metadata_filter"] = metadata_filter
        return []

    monkeypatch.setattr(
        "retrieval.conversation_retriever.rewrite_question",
        fake_rewrite_question,
    )
    monkeypatch.setattr(
        "retrieval.conversation_retriever.retrieve_documents",
        fake_retrieve_documents,
    )

    metadata_filter = {"document_id": "benefits"}

    retrieve_with_conversation(
        vector_store=mock_vector_store,
        question="What is covered?",
        top_k=5,
        metadata_filter=metadata_filter,
    )

    assert captured["top_k"] == 5
    assert captured["metadata_filter"] == metadata_filter


@pytest.mark.parametrize("question", ["", " ", None])
def test_invalid_question_raises_error(
    mock_vector_store, question
):
    with pytest.raises(ValueError):
        retrieve_with_conversation(
            vector_store=mock_vector_store,
            question=question,
        )


def test_rewriter_receives_injected_llm(
    monkeypatch, mock_vector_store
):
    fake_llm = Mock()
    captured = {}

    def fake_rewrite_question(question, chat_history, llm):
        captured["llm"] = llm
        return question

    monkeypatch.setattr(
        "retrieval.conversation_retriever.rewrite_question",
        fake_rewrite_question,
    )
    monkeypatch.setattr(
        "retrieval.conversation_retriever.retrieve_documents",
        lambda **kwargs: [],
    )

    retrieve_with_conversation(
        vector_store=mock_vector_store,
        question="What is covered?",
        llm=fake_llm,
    )

    assert captured["llm"] is fake_llm
