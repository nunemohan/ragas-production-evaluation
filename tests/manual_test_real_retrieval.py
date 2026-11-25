
from retrieval.retriever import create_retriever, retrieve_documents

VECTOR_STORE_DIR = "data/vector_store"

def main():
    print("Loading FAISS vector store...")

    vector_store = create_retriever(VECTOR_STORE_DIR)

    question = "What is the post-hospitalization coverage period?"

    print(f"\nQuestion: {question}")

    documents = retrieve_documents(
        vector_store=vector_store,
        query=question,
        top_k=3,
    )

    print(f"\nRetrieved {len(documents)} documents\n")

    for index, document in enumerate(documents, start=1):
        print(f"--- Result {index} ---")
        print("Content:", document.page_content)
        print("Metadata:", document.metadata)
        print()

if __name__ == "__main__":
    main()
