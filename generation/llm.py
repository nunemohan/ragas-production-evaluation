
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


DEFAULT_CHAT_MODEL = "gpt-4o-mini"


def create_chat_model(
    model_name: str | None = None,
    temperature: float = 0.0,
) -> ChatOpenAI:
    """
    Create and return an OpenAI chat model.

    The API key is loaded from the environment.
    """

    if not isinstance(temperature, (int, float)) or isinstance(
        temperature, bool
    ):
        raise ValueError("Temperature must be a number")

    if not 0.0 <= temperature <= 2.0:
        raise ValueError("Temperature must be between 0 and 2")

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key or not api_key.strip():
        raise ValueError("OPENAI_API_KEY is missing")

    if model_name is None:
        model_name = os.getenv("CHAT_MODEL", DEFAULT_CHAT_MODEL)

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("Model name must be a non-empty string")

    return ChatOpenAI(
        model=model_name.strip(),
        temperature=temperature,
        api_key=api_key.strip(),
    )
