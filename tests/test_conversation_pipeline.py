
import pytest
from unittest.mock import Mock

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

from generation.conversation_manager import ConversationManager
from generation.conversation_pipeline import ConversationPipeline


@pytest.fixture
def mock_vector_store():
    return Mock()


@pytest.fixture
def conversation_manager():
    return ConversationManager()


@pytest.fixture
def mock_documents():
    return [
        Document(
            page_content="Post-hospitalization expenses are covered for 60 days.",
            metadata={
                "document_id": "benefits",
                "source_filename": "02_benefits_and_limits.md",
                "chunk_index": 0,
            },
        )
    ]


@pytest.fixture
def pipeline(
    mock_vector_store,
    conversation_manager,
):
    return ConversationPipeline(
        vector_store=mock_vector_store,
        conversation_manager=conversation_manager,
    )


def setup_pipeline_mocks(
    monkeypatch,
    mock_documents,
    rewritten_question="What are the post-hospitalization benefits?",
    answer="Post-hospitalization expenses are covered for 60 days.",
):
    captured = {}

    def fake_retrieve_with_conversation(**kwargs):
        captured["retrieval"] = kwargs

        return {
            "original_question": kwargs["question"],
            "rewritten_question": rewritten_question,
            "documents": mock_documents,
        }

    def fake_build_context(documents):
        captured["context_documents"] = documents
        return "Post-hospitalization expenses are covered for 60 days."

    def fake_generate_answer(question, context, llm):
        captured["answer_question"] = question
        captured["answer_context"] = context
        captured["answer_llm"] = llm
        return answer

    monkeypatch.setattr(
        "generation.conversation_pipeline.retrieve_with_conversation",
        fake_retrieve_with_conversation,
    )

    monkeypatch.setattr(
        "generation.conversation_pipeline.build_context",
        fake_build_context,
    )

    monkeypatch.setattr(
        "generation.conversation_pipeline.generate_answer",
        fake_generate_answer,
    )

    return captured


def test_conversation_pipeline_returns_expected_result(
    monkeypatch,
    pipeline,
    mock_documents,
):
    setup_pipeline_mocks(monkeypatch, mock_documents)

    result = pipeline.ask(
        session_id="session1",
        question="What about after discharge?",
    )

    assert result["session_id"] == "session1"
    assert result["original_question"] == "What about after discharge?"
    assert result["rewritten_question"] == (
        "What are the post-hospitalization benefits?"
    )
    assert result["answer"] == (
        "Post-hospitalization expenses are covered for 60 days."
    )
    assert result["documents"] == mock_documents
    assert result["citations"] == [
    {
        "document_id": "benefits",
        "source": "02_benefits_and_limits.md",
        "chunk_index": 0,
    }
    ]


def test_conversation_history_is_saved(
    monkeypatch,
    pipeline,
    conversation_manager,
    mock_documents,
):
    setup_pipeline_mocks(monkeypatch, mock_documents)

    pipeline.ask(
        session_id="session1",
        question="What are the benefits?",
    )

    history = conversation_manager.get_history("session1")

    assert len(history) == 2
    assert isinstance(history[0], HumanMessage)
    assert isinstance(history[1], AIMessage)

    assert history[0].content == "What are the benefits?"
    assert history[1].content == (
        "Post-hospitalization expenses are covered for 60 days."
    )


def test_previous_history_is_passed_to_retriever(
    monkeypatch,
    pipeline,
    conversation_manager,
    mock_documents,
):
    conversation_manager.add_user_message(
        "session1",
        "What are hospitalization benefits?",
    )
    conversation_manager.add_ai_message(
        "session1",
        "Pre and post-hospitalization expenses are covered.",
    )

    captured = setup_pipeline_mocks(monkeypatch, mock_documents)

    pipeline.ask(
        session_id="session1",
        question="What about after discharge?",
    )

    history = captured["retrieval"]["chat_history"]

    assert len(history) == 2
    assert history[0].content == "What are hospitalization benefits?"
    assert history[1].content == (
        "Pre and post-hospitalization expenses are covered."
    )


def test_sessions_are_isolated(
    monkeypatch,
    pipeline,
    conversation_manager,
    mock_documents,
):
    setup_pipeline_mocks(monkeypatch, mock_documents)

    pipeline.ask("session1", "Question from user one")
    pipeline.ask("session2", "Question from user two")

    history1 = conversation_manager.get_history("session1")
    history2 = conversation_manager.get_history("session2")

    assert len(history1) == 2
    assert len(history2) == 2

    assert history1[0].content == "Question from user one"
    assert history2[0].content == "Question from user two"


def test_citation_builder_is_called(
    monkeypatch,
    mock_vector_store,
    conversation_manager,
    mock_documents,
):
    setup_pipeline_mocks(monkeypatch, mock_documents)

    expected_citations = [
        {
            "document_id": "benefits",
            "source": "02_benefits_and_limits.md",
        }
    ]

    captured = {}

    def fake_citation_builder(documents):
        captured["documents"] = documents
        return expected_citations

    pipeline = ConversationPipeline(
        vector_store=mock_vector_store,
        conversation_manager=conversation_manager,
        citation_builder=fake_citation_builder,
    )

    result = pipeline.ask(
        session_id="session1",
        question="What is covered?",
    )

    assert captured["documents"] == mock_documents
    assert result["citations"] == expected_citations


def test_injected_llm_is_passed_to_components(
    monkeypatch,
    mock_vector_store,
    conversation_manager,
    mock_documents,
):
    fake_llm = Mock()

    captured = setup_pipeline_mocks(monkeypatch, mock_documents)

    pipeline = ConversationPipeline(
        vector_store=mock_vector_store,
        conversation_manager=conversation_manager,
        llm=fake_llm,
    )

    pipeline.ask("session1", "What is covered?")

    assert captured["retrieval"]["llm"] is fake_llm
    assert captured["answer_llm"] is fake_llm


@pytest.mark.parametrize(
    "session_id, question",
    [
        ("", "What is covered?"),
        ("session1", ""),
        (" ", "What is covered?"),
        ("session1", " "),
    ],
)
def test_invalid_input_raises_error(
    pipeline,
    session_id,
    question,
):
    with pytest.raises(ValueError):
        pipeline.ask(
            session_id=session_id,
            question=question,
        )


def test_pipeline_does_not_save_history_when_generation_fails(
    monkeypatch,
    pipeline,
    conversation_manager,
    mock_documents,
):
    setup_pipeline_mocks(monkeypatch, mock_documents)

    def failing_generator(**kwargs):
        raise RuntimeError("LLM generation failed")

    monkeypatch.setattr(
        "generation.conversation_pipeline.generate_answer",
        failing_generator,
    )

    with pytest.raises(RuntimeError):
        pipeline.ask(
            session_id="session1",
            question="What is covered?",
        )

    assert conversation_manager.get_history("session1") == []


def test_top_k_and_metadata_filter_are_passed(
    monkeypatch,
    mock_vector_store,
    conversation_manager,
    mock_documents,
):
    captured = setup_pipeline_mocks(monkeypatch, mock_documents)

    metadata_filter = {"document_id": "benefits"}

    pipeline = ConversationPipeline(
        vector_store=mock_vector_store,
        conversation_manager=conversation_manager,
        top_k=5,
        metadata_filter=metadata_filter,
    )

    pipeline.ask("session1", "What is covered?")

    assert captured["retrieval"]["top_k"] == 5
    assert captured["retrieval"]["metadata_filter"] == metadata_filter
