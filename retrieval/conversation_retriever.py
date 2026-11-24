
"""
Conversation-aware retrieval for the RAG application.

Rewrites follow-up questions into standalone questions
before retrieving relevant documents.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document

from generation.question_rewriter import rewrite_question
from retrieval.retriever import retrieve_documents


def retrieve_with_conversation(
    vector_store,
    question: str,
    chat_history: list[BaseMessage] | None = None,
    top_k: int = 3,
    metadata_filter: dict | None = None,
    llm: BaseChatModel | None = None,
) -> tuple[str, list[Document]]:
    """
    Rewrite a follow-up question and retrieve relevant documents.

    Returns:
        A tuple containing:
        1. The standalone question used for retrieval.
        2. The retrieved document chunks.
    """

    if vector_store is None:
        raise ValueError("Vector store must not be None")

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string")

    # Convert follow-up into a standalone question.
    standalone_question = rewrite_question(
        question=question,
        chat_history=chat_history,
        llm=llm,
    )

    # Retrieve documents using the rewritten question.
    documents = retrieve_documents(
        vector_store=vector_store,
        query=standalone_question,
        top_k=top_k,
        metadata_filter=metadata_filter,
    )

    return standalone_question, documents
