
import pytest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from generation.conversation_manager import ConversationManager
from generation.conversational_rag import run_conversational_rag


@pytest.fixture
def manager():
    return ConversationManager()


@pytest.fixture
def vector_store():
    return object()


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="Post-hospitalization expenses are covered for 60 days.",
            metadata={"document_id": "benefits"},
        )
    ]


@patch("generation.conversational_rag.generate_answer")
@patch("generation.conversational_rag.build_context")
@patch("generation.conversational_rag.retrieve_with_conversation")
def test_successful_conversational_rag(
    mock_retrieve,
    mock_context,
    mock_generate,
    manager,
    vector_store,
    sample_documents,
):
    mock_retrieve.return_value = (
        "What are post-hospitalization benefits?",
        sample_documents,
    )
    mock_context.return_value = "Retrieved policy context"
    mock_generate.return_value = "Expenses are covered for 60 days."

    llm = FakeListChatModel(responses=["unused"])

    result = run_conversational_rag(
        vector_store=vector_store,
        session_id="session1",
        question="What about after discharge?",
        conversation_manager=manager,
        llm=llm,
    )

    assert result["original_question"] == "What about after discharge?"
    assert result["standalone_question"] == (
        "What are post-hospitalization benefits?"
    )
    assert result["answer"] == "Expenses are covered for 60 days."
    assert result["source_documents"] == sample_documents

    mock_generate.assert_called_once_with(
        question="What about after discharge?",
        context="Retrieved policy context",
        llm=llm,
    )

    history = manager.get_history("session1")
    assert len(history) == 2
    assert isinstance(history[0], HumanMessage)
    assert isinstance(history[1], AIMessage)


@patch("generation.conversational_rag.generate_answer")
@patch("generation.conversational_rag.build_context")
@patch("generation.conversational_rag.retrieve_with_conversation")
def test_passes_existing_history_to_retrieval(
    mock_retrieve,
    mock_context,
    mock_generate,
    manager,
    vector_store,
):
    manager.add_user_message("session1", "What are hospitalization benefits?")
    manager.add_ai_message("session1", "Pre and post-hospitalization expenses are covered.")

    expected_history = manager.get_history("session1")

    mock_retrieve.return_value = ("Standalone question", [])
    mock_context.return_value = "Some context"
    mock_generate.return_value = "Answer"

    run_conversational_rag(
        vector_store=vector_store,
        session_id="session1",
        question="What about after discharge?",
        conversation_manager=manager,
        llm=FakeListChatModel(responses=["unused"]),
    )

    mock_retrieve.assert_called_once_with(
        vector_store=vector_store,
        question="What about after discharge?",
        chat_history=expected_history,
        top_k=3,
        metadata_filter=None,
        llm=mock_generate.call_args.kwargs["llm"],
    )


@patch("generation.conversational_rag.generate_answer")
@patch("generation.conversational_rag.build_context")
@patch("generation.conversational_rag.retrieve_with_conversation")
def test_forwards_top_k_and_metadata_filter(
    mock_retrieve,
    mock_context,
    mock_generate,
    manager,
    vector_store,
):
    mock_retrieve.return_value = ("Standalone question", [])
    mock_context.return_value = "Context"
    mock_generate.return_value = "Answer"

    metadata_filter = {"document_id": "benefits"}
    llm = FakeListChatModel(responses=["unused"])

    run_conversational_rag(
        vector_store=vector_store,
        session_id="session1",
        question="What is covered?",
        conversation_manager=manager,
        top_k=5,
        metadata_filter=metadata_filter,
        llm=llm,
    )

    mock_retrieve.assert_called_once_with(
        vector_store=vector_store,
        question="What is covered?",
        chat_history=[],
        top_k=5,
        metadata_filter=metadata_filter,
        llm=llm,
    )


@patch("generation.conversational_rag.create_chat_model")
@patch("generation.conversational_rag.generate_answer")
@patch("generation.conversational_rag.build_context")
@patch("generation.conversational_rag.retrieve_with_conversation")
def test_creates_llm_when_not_provided(
    mock_retrieve,
    mock_context,
    mock_generate,
    mock_create_llm,
    manager,
    vector_store,
):
    llm = FakeListChatModel(responses=["unused"])
    mock_create_llm.return_value = llm
    mock_retrieve.return_value = ("Standalone question", [])
    mock_context.return_value = "Context"
    mock_generate.return_value = "Answer"

    run_conversational_rag(
        vector_store=vector_store,
        session_id="session1",
        question="What is covered?",
        conversation_manager=manager,
    )

    mock_create_llm.assert_called_once()
    mock_retrieve.assert_called_once()
    assert mock_retrieve.call_args.kwargs["llm"] is llm
    assert mock_generate.call_args.kwargs["llm"] is llm


@patch("generation.conversational_rag.generate_answer")
@patch("generation.conversational_rag.build_context")
@patch("generation.conversational_rag.retrieve_with_conversation")
def test_does_not_save_messages_if_generation_fails(
    mock_retrieve,
    mock_context,
    mock_generate,
    manager,
    vector_store,
):
    mock_retrieve.return_value = ("Standalone question", [])
    mock_context.return_value = "Context"
    mock_generate.side_effect = RuntimeError("Generation failed")

    with pytest.raises(RuntimeError, match="Generation failed"):
        run_conversational_rag(
            vector_store=vector_store,
            session_id="session1",
            question="What is covered?",
            conversation_manager=manager,
            llm=FakeListChatModel(responses=["unused"]),
        )

    assert manager.get_history("session1") == []


@patch("generation.conversational_rag.generate_answer")
@patch("generation.conversational_rag.build_context")
@patch("generation.conversational_rag.retrieve_with_conversation")
def test_sessions_remain_isolated(
    mock_retrieve,
    mock_context,
    mock_generate,
    manager,
    vector_store,
):
    mock_retrieve.return_value = ("Standalone question", [])
    mock_context.return_value = "Context"
    mock_generate.return_value = "Answer"

    llm = FakeListChatModel(responses=["unused"])

    for session_id in ["session1", "session2"]:
        run_conversational_rag(
            vector_store=vector_store,
            session_id=session_id,
            question=f"Question from {session_id}",
            conversation_manager=manager,
            llm=llm,
        )

    assert len(manager.get_history("session1")) == 2
    assert len(manager.get_history("session2")) == 2
    assert manager.get_history("session1")[0].content == "Question from session1"
    assert manager.get_history("session2")[0].content == "Question from session2"


def test_none_vector_store_raises_error(manager):
    with pytest.raises(ValueError, match="Vector store"):
        run_conversational_rag(
            vector_store=None,
            session_id="session1",
            question="What is covered?",
            conversation_manager=manager,
        )


def test_invalid_question_raises_error(manager, vector_store):
    with pytest.raises(ValueError, match="Question"):
        run_conversational_rag(
            vector_store=vector_store,
            session_id="session1",
            question=" ",
            conversation_manager=manager,
        )


def test_invalid_session_id_raises_error(manager, vector_store):
    with pytest.raises(ValueError, match="Session ID"):
        run_conversational_rag(
            vector_store=vector_store,
            session_id=" ",
            question="What is covered?",
            conversation_manager=manager,
        )


def test_invalid_conversation_manager_raises_error(vector_store):
    with pytest.raises(ValueError, match="ConversationManager"):
        run_conversational_rag(
            vector_store=vector_store,
            session_id="session1",
            question="What is covered?",
            conversation_manager=None,
        )
