
import pytest

from generation.hallucination_guard import (
    ABSTENTION_MESSAGE,
    NO_CONTEXT_MESSAGE,
    has_retrieved_evidence,
    should_abstain,
    get_abstention_response,
    is_abstention_response,
)


def test_has_retrieved_evidence_with_valid_context():
    context = "[Context 1]\nContent:\nCoverage is 30 days."

    assert has_retrieved_evidence(context) is True


def test_has_retrieved_evidence_with_empty_context():
    assert has_retrieved_evidence("") is False


def test_has_retrieved_evidence_with_whitespace():
    assert has_retrieved_evidence("   ") is False


def test_has_retrieved_evidence_with_no_context_message():
    assert has_retrieved_evidence(NO_CONTEXT_MESSAGE) is False


def test_should_abstain_when_context_is_unavailable():
    assert should_abstain("") is True


def test_should_not_abstain_when_context_is_available():
    context = "Pre-hospitalization expenses are covered for 30 days."

    assert should_abstain(context) is False


def test_get_abstention_response_when_context_is_missing():
    assert get_abstention_response("") == ABSTENTION_MESSAGE


def test_get_abstention_response_when_context_is_available():
    context = "Coverage period is 30 days."

    assert get_abstention_response(context) is None


def test_is_abstention_response():
    assert is_abstention_response(ABSTENTION_MESSAGE) is True


def test_is_not_abstention_response():
    assert is_abstention_response("Coverage is 30 days.") is False


@pytest.mark.parametrize("invalid_context", [None, 123, [], {}])
def test_invalid_context_raises_error(invalid_context):
    with pytest.raises(ValueError, match="Context must be a string"):
        has_retrieved_evidence(invalid_context)


def test_invalid_answer_raises_error():
    with pytest.raises(ValueError, match="Answer must be a string"):
        is_abstention_response(None)
