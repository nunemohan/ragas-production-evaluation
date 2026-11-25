
from retrieval.retriever import create_retriever

from generation.conversation_manager import ConversationManager
from generation.conversation_pipeline import ConversationPipeline


def main():
    print("Loading FAISS vector store...")

    vector_store = create_retriever("data/vector_store")

    conversation_manager = ConversationManager()

    pipeline = ConversationPipeline(
        vector_store=vector_store,
        conversation_manager=conversation_manager,
        top_k=3,
    )

    question = "What is the post-hospitalization coverage period?"

    print("\nUser question:", question)
    print("\nGenerating answer...\n")

    result = pipeline.ask(
        session_id="real_test_session",
        question=question,
    )

    print("=" * 60)
    print("CONVERSATIONAL RAG RESULT")
    print("=" * 60)

    print("\nOriginal question:")
    print(result["original_question"])

    print("\nRewritten question:")
    print(result["rewritten_question"])

    print("\nGenerated answer:")
    print(result["answer"])

    print("\nSource citations:")
    for citation in result["citations"]:
        print(citation)

    print("\nConversation history:")
    for message in conversation_manager.get_history(
        "real_test_session"
    ):
        print(type(message).__name__, ":", message.content)


if __name__ == "__main__":
    main()
