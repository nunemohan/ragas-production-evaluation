
import os
import uuid

import pytest

from retrieval.retriever import create_retriever
from generation.conversation_manager import ConversationManager
from generation.conversation_pipeline import ConversationPipeline
from generation.hallucination_guard import ABSTENTION_MESSAGE


RUN_LIVE_TESTS = os.getenv("RUN_LIVE_RAG_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_LIVE_TESTS,
    reason="Set RUN_LIVE_RAG_TESTS=1 to execute live RAG integration tests",
)


@pytest.fixture(scope="module")
def pipeline():
    vector_store = create_retriever("data/vector_store")

    manager = ConversationManager()

    return ConversationPipeline(
        vector_store=vector_store,
        conversation_manager=manager,
        top_k=3,
    )


def test_supported_question_end_to_end(pipeline):
    session_id = f"supported-{uuid.uuid4().hex[:8]}"

    result = pipeline.ask(
        session_id=session_id,
        question="What is the post-hospitalization coverage period?",
    )

    assert "60 days" in result["answer"].lower()
    assert result["documents"]
    assert result["citations"]

    history = pipeline.conversation_manager.get_history(session_id)

    assert len(history) == 2


def test_unsupported_question_end_to_end(pipeline):
    session_id = f"unsupported-{uuid.uuid4().hex[:8]}"

    result = pipeline.ask(
        session_id=session_id,
        question="Is planned overseas treatment covered under this policy?",
    )

    assert result["answer"] == ABSTENTION_MESSAGE
    assert result["documents"]

    history = pipeline.conversation_manager.get_history(session_id)

    assert len(history) == 2


def test_multi_turn_conversation_end_to_end(pipeline):
    session_id = f"multi-turn-{uuid.uuid4().hex[:8]}"

    # Turn 1: Initial question
    result1 = pipeline.ask(
        session_id=session_id,
        question="What is the pre-hospitalization coverage period?",
    )

    assert "30 days" in result1["answer"].lower()
    assert result1["citations"]

    # Turn 2: Follow-up question
    result2 = pipeline.ask(
        session_id=session_id,
        question="What about after discharge?",
    )

    assert "60 days" in result2["answer"].lower()
    assert "discharge" in result2["rewritten_question"].lower()
    assert result2["rewritten_question"] != "What about after discharge?"
    assert result2["citations"]

    # Turn 3: Another question in the same conversation
    result3 = pipeline.ask(
        session_id=session_id,
        question="How long is the waiting period for pre-existing diseases?",
    )

    assert "36 months" in result3["answer"].lower()
    assert result3["citations"]

    # Validate conversation history
    history = pipeline.conversation_manager.get_history(session_id)

    assert len(history) == 6

    assert history[0].type == "human"
    assert history[1].type == "ai"

    assert history[2].type == "human"
    assert history[3].type == "ai"

    assert history[4].type == "human"
    assert history[5].type == "ai"

    # Validate all three user questions were stored
    assert history[0].content == (
        "What is the pre-hospitalization coverage period?"
    )
    assert history[2].content == "What about after discharge?"
    assert history[4].content == (
        "How long is the waiting period for pre-existing diseases?"
    )
