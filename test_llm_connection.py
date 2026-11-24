
from generation.llm import create_chat_model


def main():
    chat_model = create_chat_model()

    response = chat_model.invoke(
        "Explain RAG in two simple sentences."
    )

    print("\n========== LLM RESPONSE ==========")
    print(response.content)

    print("\n========== TOKEN USAGE ==========")
    print(response.usage_metadata)


if __name__ == "__main__":
    main()
