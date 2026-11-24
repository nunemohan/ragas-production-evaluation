
import pytest
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from generation.question_rewriter import (
    get_question_rewrite_prompt,
    rewrite_question,
)


def test_rewrite_prompt_has_expected_variables():
    prompt = get_question_rewrite_prompt()

    assert "chat_history" in prompt.input_variables
    assert "question" in prompt.input_variables


def test_rewrite_follow_up_question():
    llm = FakeListChatModel(
        responses=["What are the post-hospitalization benefits?"]
    )

    history = [
        HumanMessage(content="What are the hospitalization benefits?"),
        AIMessage(content="Pre and post-hospitalization expenses are covered."),
    ]

    result = rewrite_question(
        question="What about after discharge?",
        chat_history=history,
        llm=llm,
    )

    assert result == "What are the post-hospitalization benefits?"


def test_rewrite_strips_whitespace():
    llm = FakeListChatModel(responses=["  What is the waiting period?  "])

    history = [HumanMessage(content="Tell me about waiting periods.")]

    result = rewrite_question(
        "What about that?",
        chat_history=history,
        llm=llm,
    )

    assert result == "What is the waiting period?"


def test_no_history_returns_original_question():
    result = rewrite_question("What is the coverage?")

    assert result == "What is the coverage?"


def test_empty_question_raises_error():
    with pytest.raises(ValueError):
        rewrite_question("")


def test_invalid_history_type():
    with pytest.raises(ValueError):
        rewrite_question("What about that?", chat_history="invalid")


def test_invalid_history_message():
    with pytest.raises(ValueError):
        rewrite_question("What about that?", chat_history=["invalid"])


def test_empty_llm_response_raises_error():
    llm = FakeListChatModel(responses=["   "])

    history = [HumanMessage(content="Tell me about waiting periods.")]

    with pytest.raises(ValueError):
        rewrite_question(
            "What about that?",
            chat_history=history,
            llm=llm,
        )
