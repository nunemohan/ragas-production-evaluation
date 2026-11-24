
from langchain_core.language_models import BaseChatModel

from generation.llm import create_chat_model
from generation.prompt_template import get_rag_prompt
from generation.hallucination_guard import get_abstention_response


def generate_answer(
    question: str,
    context: str,
    llm: BaseChatModel | None = None,
) -> str:
    """
    Generate an answer using the grounded RAG prompt.

    Abstains immediately when retrieved context is unavailable.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string")

    # Stage 4.6: Check whether retrieved evidence is available.
    abstention_response = get_abstention_response(context)

    if abstention_response is not None:
        return abstention_response

    # Create the LLM only when usable context is available.
    if llm is None:
        llm = create_chat_model()

    prompt = get_rag_prompt()

    chain = prompt | llm

    response = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    if not isinstance(response.content, str):
        raise TypeError("LLM response content must be a string")

    answer = response.content.strip()

    if not answer:
        raise ValueError("LLM returned an empty response")

    return answer
