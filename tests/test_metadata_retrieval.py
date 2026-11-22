
from pathlib import Path

from retrieval.retriever import create_retriever, retrieve_documents


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"


def main():
    query = "What documents are required for reimbursement claims?"

    vector_store = create_retriever(str(VECTOR_STORE_DIR))

    results = retrieve_documents(
        vector_store=vector_store,
        query=query,
        top_k=3,
        metadata_filter={
            "document_id": "04_claims_procedure"
        },
    )

    print(f"\nQuery: {query}")
    print(f"Retrieved chunks: {len(results)}")

    for index, document in enumerate(results, start=1):
        print(f"\n--- Result {index} ---")
        print("Document ID:", document.metadata.get("document_id"))
        print("Content:", document.page_content)


if __name__ == "__main__":
    main()
