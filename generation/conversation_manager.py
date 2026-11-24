
"""
Conversation management for the RAG application.

Maintains isolated in-memory conversation histories
for different sessions.
"""

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)


class ConversationManager:
    """
    Manage conversation history using session IDs.

    History is stored in memory and is lost when the
    application process restarts.
    """

    def __init__(self, max_messages: int = 10):
        if (
            not isinstance(max_messages, int)
            or isinstance(max_messages, bool)
            or max_messages <= 0
        ):
            raise ValueError("max_messages must be a positive integer")

        self.max_messages = max_messages
        self._sessions: dict[str, list[BaseMessage]] = {}

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("Session ID must be a non-empty string")

    @staticmethod
    def _validate_content(content: str) -> None:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Message content must be a non-empty string")

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        """Add a user or assistant message to a session."""

        self._validate_session_id(session_id)
        self._validate_content(content)

        normalized_role = role.strip().lower() if isinstance(role, str) else ""

        if normalized_role == "user":
            message = HumanMessage(content=content.strip())
        elif normalized_role in ("assistant", "ai"):
            message = AIMessage(content=content.strip())
        else:
            raise ValueError("Role must be 'user' or 'assistant'")

        session_key = session_id.strip()
        history = self._sessions.setdefault(session_key, [])
        history.append(message)

        self._sessions[session_key] = history[-self.max_messages:]

    def add_user_message(self, session_id: str, content: str) -> None:
        """Add a user message."""
        self.add_message(session_id, "user", content)

    def add_ai_message(self, session_id: str, content: str) -> None:
        """Add an assistant message."""
        self.add_message(session_id, "assistant", content)

    def get_history(self, session_id: str) -> list[BaseMessage]:
        """Return a copy of the conversation history."""

        self._validate_session_id(session_id)
        return list(self._sessions.get(session_id.strip(), []))

    def clear_history(self, session_id: str) -> None:
        """Clear history for a specific session."""

        self._validate_session_id(session_id)
        self._sessions.pop(session_id.strip(), None)

    def get_session_count(self) -> int:
        """Return the number of active sessions."""
        return len(self._sessions)
