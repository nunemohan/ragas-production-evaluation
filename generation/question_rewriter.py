
"""
Question rewriting for conversational RAG.

Converts follow-up questions into standalone questions
using previous conversation history.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from generation.llm import create_chat_model


def get_question_rewrite_prompt() -> ChatPromptTemplate:
    """Create the follow-up question rewriting prompt."""

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a question rewriting assistant for an insurance
knowledge-base RAG application.

Your task is to convert the user's latest question into
a standalone question using conversation history.

Follow these rules:

1. Use conversation history to understand references.
2. Preserve the original meaning and intent.
3. Resolve references such as "it", "that", "those",
   and "what about after discharge".
4. Include relevant context from previous messages.
5. Do not answer the user's question.
6. Do not introduce unsupported facts.
7. If the question is already standalone, preserve its meaning.
8. Return only the rewritten question.
9. Do not include explanations or commentary.
                """,
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "human",
                "Latest user question:\n{question}\n\n"
                "Return the standalone question only.",
            ),
        ]
    )


def rewrite_question(
    question: str,
    chat_history: list[BaseMessage] | None = None,
    llm: BaseChatModel | None = None,
) -> str:
    """
    Rewrite a follow-up question into a standalone question.

    Returns the original question unchanged when no history exists.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string")

    if chat_history is None:
        chat_history = []

    if not isinstance(chat_history, list):
        raise ValueError("Chat history must be a list")

    if not all(isinstance(message, BaseMessage) for message in chat_history):
        raise ValueError(
            "Every chat history item must be a LangChain BaseMessage"
        )

    if not chat_history:
        return question.strip()

    if llm is None:
        llm = create_chat_model()

    prompt = get_question_rewrite_prompt()
    chain = prompt | llm

    response = chain.invoke(
        {
            "chat_history": chat_history,
            "question": question.strip(),
        }
    )

    if not isinstance(response.content, str):
        raise TypeError("LLM response content must be a string")

    rewritten_question = response.content.strip()

    if not rewritten_question:
        raise ValueError("Question rewriter returned an empty response")

    return rewritten_question
