from generation.citation_builder import build_citations

"""
End-to-end conversational RAG pipeline.

Connects conversation management, question rewriting,
document retrieval, context preparation, answer generation,
and source citations.
"""

from typing import Callable, Any

from langchain_core.language_models import BaseChatModel

from generation.conversation_manager import ConversationManager
from generation.context_builder import build_context
from generation.answer_generator import generate_answer

from retrieval.conversation_retriever import (
    retrieve_with_conversation,
)


class ConversationPipeline:
    """
    Orchestrates conversational RAG for multiple sessions.

    Dependencies are injected to make the pipeline
    independently testable.
    """

    def __init__(
        self,
        vector_store,
        conversation_manager: ConversationManager,
        llm: BaseChatModel | None = None,
        top_k: int = 3,
        metadata_filter: dict | None = None,
        citation_builder: Callable[[list], Any] | None = None,
    ):
        if vector_store is None:
            raise ValueError("Vector store must not be None")

        if conversation_manager is None:
            raise ValueError("Conversation manager must not be None")

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        if metadata_filter is not None and not isinstance(metadata_filter, dict):
            raise ValueError("metadata_filter must be a dictionary or None")

        if citation_builder is not None and not callable(citation_builder):
            raise ValueError("citation_builder must be callable or None")

        self.vector_store = vector_store
        self.conversation_manager = conversation_manager
        self.llm = llm
        self.top_k = top_k
        self.metadata_filter = metadata_filter
        self.citation_builder = citation_builder

    def ask(
        self,
        session_id: str,
        question: str,
    ) -> dict:
        """
        Process a user question within a conversation session.

        Returns:
            Dictionary containing:
            - session_id
            - original_question
            - rewritten_question
            - answer
            - documents
            - citations
        """

        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("Session ID must be a non-empty string")

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string")

        session_id = session_id.strip()
        question = question.strip()

        # Step 1: Retrieve previous conversation history.
        chat_history = self.conversation_manager.get_history(
            session_id
        )

        # Step 2: Rewrite the question and retrieve relevant documents.
        retrieval_result = retrieve_with_conversation(
            vector_store=self.vector_store,
            question=question,
            chat_history=chat_history,
            top_k=self.top_k,
            metadata_filter=self.metadata_filter,
            llm=self.llm,
        )

        rewritten_question = retrieval_result["rewritten_question"]
        documents = retrieval_result["documents"]

        # Step 3: Prepare context from retrieved documents.
        context = build_context(documents)

        # Step 4: Generate a grounded answer.
        answer = generate_answer(
            question=rewritten_question,
            context=context,
            llm=self.llm,
        )

        # Step 5: Build source citations.
        citations = (
            self.citation_builder(documents)
            if self.citation_builder is not None
            else []
        )

        # Step 6: Save the conversation after successful processing.
        self.conversation_manager.add_user_message(
            session_id,
            question,
        )

        self.conversation_manager.add_ai_message(
            session_id,
            answer,
        )

        # Step 7: Return the complete result.
        return {
            "session_id": session_id,
            "original_question": question,
            "rewritten_question": rewritten_question,
            "answer": answer,
            "documents": documents,
            "citations": citations,
        }
