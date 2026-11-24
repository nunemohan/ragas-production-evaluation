
import pytest
from langchain_core.messages import HumanMessage, AIMessage

from generation.conversation_manager import ConversationManager


def test_add_user_message():
    manager = ConversationManager()
    manager.add_user_message("session1", "Hello")

    history = manager.get_history("session1")

    assert len(history) == 1
    assert isinstance(history[0], HumanMessage)
    assert history[0].content == "Hello"


def test_add_ai_message():
    manager = ConversationManager()
    manager.add_ai_message("session1", "Hello! How can I help?")

    history = manager.get_history("session1")

    assert len(history) == 1
    assert isinstance(history[0], AIMessage)


def test_add_multiple_messages():
    manager = ConversationManager()
    manager.add_user_message("session1", "What is the coverage?")
    manager.add_ai_message("session1", "The coverage is ₹5 lakh.")

    history = manager.get_history("session1")

    assert len(history) == 2
    assert isinstance(history[0], HumanMessage)
    assert isinstance(history[1], AIMessage)


def test_sessions_are_isolated():
    manager = ConversationManager()
    manager.add_user_message("session1", "Question A")
    manager.add_user_message("session2", "Question B")

    assert manager.get_history("session1")[0].content == "Question A"
    assert manager.get_history("session2")[0].content == "Question B"


def test_max_messages_limit():
    manager = ConversationManager(max_messages=2)
    manager.add_user_message("session1", "Message 1")
    manager.add_ai_message("session1", "Message 2")
    manager.add_user_message("session1", "Message 3")

    history = manager.get_history("session1")

    assert len(history) == 2
    assert history[0].content == "Message 2"
    assert history[1].content == "Message 3"


def test_clear_history():
    manager = ConversationManager()
    manager.add_user_message("session1", "Hello")
    manager.clear_history("session1")

    assert manager.get_history("session1") == []


def test_get_history_returns_copy():
    manager = ConversationManager()
    manager.add_user_message("session1", "Hello")

    history = manager.get_history("session1")
    history.clear()

    assert len(manager.get_history("session1")) == 1


def test_get_session_count():
    manager = ConversationManager()
    manager.add_user_message("session1", "Hello")
    manager.add_user_message("session2", "Hi")

    assert manager.get_session_count() == 2


def test_invalid_max_messages():
    with pytest.raises(ValueError):
        ConversationManager(max_messages=0)


@pytest.mark.parametrize("session_id", ["", " ", None])
def test_invalid_session_id(session_id):
    manager = ConversationManager()

    with pytest.raises(ValueError):
        manager.get_history(session_id)


@pytest.mark.parametrize("content", ["", " ", None])
def test_invalid_message_content(content):
    manager = ConversationManager()

    with pytest.raises(ValueError):
        manager.add_user_message("session1", content)


def test_invalid_role():
    manager = ConversationManager()

    with pytest.raises(ValueError):
        manager.add_message("session1", "system", "Hello")
