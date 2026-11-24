
import pytest
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from generation.answer_generator import generate_answer
from generation.hallucination_guard import ABSTENTION_MESSAGE


def test_generate_answer_returns_llm_response():
    fake_llm = FakeListChatModel(
        responses=[
            "Pre-hospitalization expenses are covered for 30 days."
        ]
    )

    answer = generate_answer(
        question="What is the pre-hospitalization coverage period?",
        context="Pre-hospitalization expenses are covered for 30 days.",
        llm=fake_llm,
    )

    assert answer == (
        "Pre-hospitalization expenses are covered for 30 days."
    )


def test_generate_answer_strips_whitespace():
    fake_llm = FakeListChatModel(
        responses=["  Coverage is 30 days.  "]
    )

    answer = generate_answer(
        question="What is the coverage period?",
        context="Coverage period is 30 days.",
        llm=fake_llm,
    )

    assert answer == "Coverage is 30 days."


def test_generate_answer_rejects_empty_question():
    with pytest.raises(ValueError, match="Question"):
        generate_answer(
            question=" ",
            context="Some policy context.",
            llm=FakeListChatModel(responses=["Answer"]),
        )


def test_generate_answer_creates_default_llm():
    fake_llm = FakeListChatModel(
        responses=["The coverage period is 30 days."]
    )

    with patch(
        "generation.answer_generator.create_chat_model",
        return_value=fake_llm,
    ) as mock_create:

        answer = generate_answer(
            question="What is the coverage period?",
            context="Coverage period is 30 days.",
        )

    mock_create.assert_called_once_with()
    assert answer == "The coverage period is 30 days."


def test_generate_answer_rejects_empty_llm_response():
    fake_llm = FakeListChatModel(responses=["   "])

    with pytest.raises(ValueError, match="empty response"):
        generate_answer(
            question="What is covered?",
            context="Some policy context.",
            llm=fake_llm,
        )


def test_generate_answer_abstains_when_context_is_empty():
    with patch(
        "generation.answer_generator.create_chat_model"
    ) as mock_create:

        answer = generate_answer(
            question="Is home nursing covered?",
            context="",
        )

    assert answer == ABSTENTION_MESSAGE
    mock_create.assert_not_called()


def test_generate_answer_abstains_when_context_is_unavailable():
    with patch(
        "generation.answer_generator.create_chat_model"
    ) as mock_create:

        answer = generate_answer(
            question="Is home nursing covered?",
            context="No relevant context was retrieved.",
        )

    assert answer == ABSTENTION_MESSAGE
    mock_create.assert_not_called()
