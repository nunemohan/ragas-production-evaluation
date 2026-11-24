
import pytest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

from retrieval.conversation_retriever import retrieve_with_conversation


@pytest.fixture
def mock_vector_store():
    return object()


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="Post-hospitalization expenses are covered for 60 days.",
            metadata={"document_id": "benefits"},
        )
    ]


@patch("retrieval.conversation_retriever.retrieve_documents")
@patch("retrieval.conversation_retriever.rewrite_question")
def test_retrieves_using_rewritten_question(
    mock_rewrite,
    mock_retrieve,
    mock_vector_store,
    sample_documents,
):
    mock_rewrite.return_value = "What are post-hospitalization benefits?"
    mock_retrieve.return_value = sample_documents

    history = [
        HumanMessage(content="What are hospitalization benefits?"),
        AIMessage(content="Pre and post-hospitalization expenses are covered."),
    ]

    standalone_question, documents = retrieve_with_conversation(
        vector_store=mock_vector_store,
        question="What about after discharge?",
        chat_history=history,
    )

    mock_rewrite.assert_called_once_with(
        question="What about after discharge?",
        chat_history=history,
        llm=None,
    )

    mock_retrieve.assert_called_once_with(
        vector_store=mock_vector_store,
        query="What are post-hospitalization benefits?",
        top_k=3,
        metadata_filter=None,
    )

    assert standalone_question == "What are post-hospitalization benefits?"
    assert documents == sample_documents


@patch("retrieval.conversation_retriever.retrieve_documents")
@patch("retrieval.conversation_retriever.rewrite_question")
def test_retrieves_without_history(
    mock_rewrite,
    mock_retrieve,
    mock_vector_store,
    sample_documents,
):
    mock_rewrite.return_value = "What is the coverage?"
    mock_retrieve.return_value = sample_documents

    standalone_question, documents = retrieve_with_conversation(
        vector_store=mock_vector_store,
        question="What is the coverage?",
    )

    mock_rewrite.assert_called_once_with(
        question="What is the coverage?",
        chat_history=None,
        llm=None,
    )

    assert standalone_question == "What is the coverage?"
    assert documents == sample_documents


@patch("retrieval.conversation_retriever.retrieve_documents")
@patch("retrieval.conversation_retriever.rewrite_question")
def test_forwards_top_k_and_metadata_filter(
    mock_rewrite,
    mock_retrieve,
    mock_vector_store,
):
    mock_rewrite.return_value = "Standalone question"
    mock_retrieve.return_value = []

    metadata_filter = {"document_id": "benefits"}

    retrieve_with_conversation(
        vector_store=mock_vector_store,
        question="What about that?",
        top_k=5,
        metadata_filter=metadata_filter,
    )

    mock_retrieve.assert_called_once_with(
        vector_store=mock_vector_store,
        query="Standalone question",
        top_k=5,
        metadata_filter=metadata_filter,
    )


@patch("retrieval.conversation_retriever.retrieve_documents")
@patch("retrieval.conversation_retriever.rewrite_question")
def test_forwards_llm(
    mock_rewrite,
    mock_retrieve,
    mock_vector_store,
):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    llm = FakeListChatModel(responses=["Standalone question"])
    mock_rewrite.return_value = "Standalone question"
    mock_retrieve.return_value = []

    retrieve_with_conversation(
        vector_store=mock_vector_store,
        question="What about that?",
        llm=llm,
    )

    mock_rewrite.assert_called_once_with(
        question="What about that?",
        chat_history=None,
        llm=llm,
    )


@pytest.mark.parametrize("question", ["", " ", None])
def test_invalid_question_raises_error(mock_vector_store, question):
    with pytest.raises(ValueError):
        retrieve_with_conversation(
            vector_store=mock_vector_store,
            question=question,
        )


def test_none_vector_store_raises_error():
    with pytest.raises(ValueError):
        retrieve_with_conversation(
            vector_store=None,
            question="What is the coverage?",
        )


@patch("retrieval.conversation_retriever.retrieve_documents")
@patch("retrieval.conversation_retriever.rewrite_question")
def test_original_question_is_not_modified(
    mock_rewrite,
    mock_retrieve,
    mock_vector_store,
):
    original_question = "What about after discharge?"
    mock_rewrite.return_value = "What are post-hospitalization benefits?"
    mock_retrieve.return_value = []

    standalone_question, _ = retrieve_with_conversation(
        vector_store=mock_vector_store,
        question=original_question,
    )

    assert original_question == "What about after discharge?"
    assert standalone_question != original_question
