
from generation.prompt_template import get_rag_prompt


def test_get_rag_prompt_returns_chat_prompt_template():
    from langchain_core.prompts import ChatPromptTemplate

    prompt = get_rag_prompt()

    assert isinstance(prompt, ChatPromptTemplate)


def test_prompt_contains_required_variables():
    prompt = get_rag_prompt()

    assert set(prompt.input_variables) == {"context", "question"}


def test_prompt_formats_context_and_question():
    prompt = get_rag_prompt()

    formatted_prompt = prompt.format_messages(
        context="Pre-hospitalization expenses are covered for 30 days.",
        question="What is the pre-hospitalization coverage period?",
    )

    assert len(formatted_prompt) == 2
    assert "30 days" in formatted_prompt[0].content
    assert "What is the pre-hospitalization coverage period?" in formatted_prompt[1].content


def test_system_prompt_contains_grounding_instructions():
    prompt = get_rag_prompt()

    formatted_prompt = prompt.format_messages(
        context="Sample policy context.",
        question="Sample question?",
    )

    system_content = formatted_prompt[0].content

    assert "Use only the information provided in the context" in system_content
    assert "Do not invent facts" in system_content


def test_prompt_contains_insufficient_information_instruction():
    prompt = get_rag_prompt()

    formatted_prompt = prompt.format_messages(
        context="No relevant context was retrieved.",
        question="Is home nursing covered?",
    )

    assert "I don't have sufficient information" in formatted_prompt[0].content
