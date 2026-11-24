import pytest
from unittest.mock import patch

from generation.llm import create_chat_model


def test_create_chat_model_success():
    with patch("generation.llm.load_dotenv"), \
         patch.dict(
             "os.environ",
             {"OPENAI_API_KEY": "test-api-key"},
             clear=True,
         ), \
         patch("generation.llm.ChatOpenAI") as mock_chat:

        create_chat_model()

        mock_chat.assert_called_once_with(
            model="gpt-4o-mini",
            temperature=0.0,
            api_key="test-api-key",
        )


def test_create_chat_model_uses_custom_model():
    with patch("generation.llm.load_dotenv"), \
         patch.dict(
             "os.environ",
             {
                 "OPENAI_API_KEY": "test-api-key",
                 "CHAT_MODEL": "custom-model",
             },
             clear=True,
         ), \
         patch("generation.llm.ChatOpenAI") as mock_chat:

        create_chat_model()

        assert mock_chat.call_args.kwargs["model"] == "custom-model"


def test_create_chat_model_missing_api_key():
    with patch("generation.llm.load_dotenv"), \
         patch.dict("os.environ", {}, clear=True):

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            create_chat_model()


@pytest.mark.parametrize("temperature", [-0.1, 2.1, True, "0.5"])
def test_create_chat_model_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="Temperature"):
        create_chat_model(temperature=temperature)


def test_create_chat_model_invalid_model_name():
    with patch("generation.llm.load_dotenv"), \
         patch.dict(
             "os.environ",
             {"OPENAI_API_KEY": "test-api-key"},
             clear=True,
         ):

        with pytest.raises(ValueError, match="Model name"):
            create_chat_model(model_name=" ")
