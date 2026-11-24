
from langchain_core.language_models import BaseChatModel

from generation.llm import create_chat_model
from generation.prompt_template import get_rag_prompt


def generate_answer(
    question: str,
    context: str,
    llm: BaseChatModel | None = None,
) -> str:
    """
    Generate an answer using the grounded RAG prompt
    and the supplied retrieved context.

    Args:
        question: User's question.
        context: Context prepared from retrieved documents.
        llm: Optional chat model. If not provided,
             the default OpenAI chat model is created.

    Returns:
        Generated answer as a string.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string")

    if not isinstance(context, str) or not context.strip():
        raise ValueError("Context must be a non-empty string")

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
