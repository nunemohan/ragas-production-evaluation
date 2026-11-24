
"""
Conversation-aware retrieval for the RAG application.

Rewrites follow-up questions into standalone questions
before retrieving relevant documents.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from generation.question_rewriter import rewrite_question
from retrieval.retriever import retrieve_documents


def retrieve_with_conversation(
    vector_store,
    question: str,
    chat_history: list[BaseMessage] | None = None,
    top_k: int = 3,
    metadata_filter: dict | None = None,
    llm: BaseChatModel | None = None,
) -> dict:
    """
    Rewrite a follow-up question and retrieve relevant documents.

    Returns:
        Dictionary containing:
        - original_question
        - rewritten_question
        - documents
    """

    if vector_store is None:
        raise ValueError("Vector store must not be None")

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string")

    original_question = question.strip()

    # Step 1: Rewrite the question using conversation history.
    rewritten_question = rewrite_question(
        question=original_question,
        chat_history=chat_history,
        llm=llm,
    )

    # Step 2: Retrieve documents using the rewritten question.
    documents = retrieve_documents(
        vector_store=vector_store,
        query=rewritten_question,
        top_k=top_k,
        metadata_filter=metadata_filter,
    )

    # Step 3: Return retrieval details.
    return {
        "original_question": original_question,
        "rewritten_question": rewritten_question,
        "documents": documents,
    }
