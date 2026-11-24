
"""
Hallucination prevention utilities for the RAG generation pipeline.

Provides deterministic checks for missing or unavailable
retrieved context and a standard abstention response.
"""

ABSTENTION_MESSAGE = (
    "I don't have sufficient information in the available "
    "policy documents to answer this question."
)

NO_CONTEXT_MESSAGE = "No relevant context was retrieved."


def has_retrieved_evidence(context: str) -> bool:
    """
    Check whether usable retrieved context is available.

    This is a structural availability check, not a semantic
    verification that the context supports the question.
    """

    if not isinstance(context, str):
        raise ValueError("Context must be a string")

    normalized_context = context.strip()

    if not normalized_context:
        return False

    if normalized_context.casefold() == NO_CONTEXT_MESSAGE.casefold():
        return False

    return True


def should_abstain(context: str) -> bool:
    """
    Determine whether the system should abstain because
    retrieved context is unavailable.
    """

    return not has_retrieved_evidence(context)


def get_abstention_response(context: str) -> str | None:
    """
    Return the standard abstention message when context
    is unavailable. Otherwise, return None.
    """

    if should_abstain(context):
        return ABSTENTION_MESSAGE

    return None


def is_abstention_response(answer: str) -> bool:
    """
    Check whether an answer matches the standard abstention message.
    """

    if not isinstance(answer, str):
        raise ValueError("Answer must be a string")

    return answer.strip().casefold() == ABSTENTION_MESSAGE.casefold()
