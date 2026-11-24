
"""
End-to-end conversational RAG pipeline.

Coordinates conversation history, question rewriting,
retrieval, context preparation, and answer generation.
"""

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from generation.conversation_manager import ConversationManager
from generation.context_builder import build_context
from generation.answer_generator import generate_answer
from generation.llm import create_chat_model

from retrieval.conversation_retriever import retrieve_with_conversation


def run_conversational_rag(
    vector_store,
    session_id: str,
    question: str,
    conversation_manager: ConversationManager,
    top_k: int = 3,
    metadata_filter: dict | None = None,
    llm: BaseChatModel | None = None,
) -> dict:
    """
    Execute one conversational RAG interaction.

    Returns:
        A dictionary containing the original question,
        standalone question, generated answer,
        and retrieved source documents.
    """

    if vector_store is None:
        raise ValueError("Vector store must not be None")

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string")

    if not isinstance(conversation_manager, ConversationManager):
        raise ValueError("A valid ConversationManager is required")

    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("Session ID must be a non-empty string")

    original_question = question.strip()
    normalized_session_id = session_id.strip()

    # 1. Load existing conversation history.
    chat_history = conversation_manager.get_history(
        normalized_session_id
    )

    # 2. Create one shared LLM instance.
    if llm is None:
        llm = create_chat_model()

    # 3. Rewrite follow-up and retrieve relevant documents.
    standalone_question, documents = retrieve_with_conversation(
        vector_store=vector_store,
        question=original_question,
        chat_history=chat_history,
        top_k=top_k,
        metadata_filter=metadata_filter,
        llm=llm,
    )

    # 4. Prepare context from retrieved documents.
    context = build_context(documents)

    # 5. Generate answer using the ORIGINAL user question.
    answer = generate_answer(
        question=original_question,
        context=context,
        llm=llm,
    )

    # 6. Save conversation only after successful generation.
    conversation_manager.add_user_message(
        normalized_session_id,
        original_question,
    )

    conversation_manager.add_ai_message(
        normalized_session_id,
        answer,
    )

    # 7. Return the complete interaction result.
    return {
        "session_id": normalized_session_id,
        "original_question": original_question,
        "standalone_question": standalone_question,
        "answer": answer,
        "source_documents": documents,
    }
