
from pathlib import Path

from retrieval.retriever import create_retriever


PROJECT_ROOT = Path(__file__).resolve().parent
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"

QUERIES = {
    "A - Hospitalization expense windows": (
        "How many days before admission and after discharge "
        "are eligible hospitalization expenses covered?"
    ),
    "B - Reimbursement documents": (
        "What documents must be submitted for a reimbursement "
        "claim after treatment at a non-network hospital?"
    ),
    "C - Original combined Q11": (
        "After a covered hospitalization at a non-network hospital, "
        "what pre- and post-hospitalization expense windows apply, "
        "and what documents should I submit for reimbursement?"
    ),
}


def main():
    vector_store = create_retriever(str(VECTOR_STORE_DIR))

    for query_name, query in QUERIES.items():
        print("\n" + "=" * 80)
        print(query_name)
        print(f"Query: {query}")
        print("=" * 80)

        results = vector_store.similarity_search_with_score(
            query=query,
            k=5,
        )

        for rank, (document, score) in enumerate(results, start=1):
            metadata = document.metadata

            print(f"\nRank: {rank}")
            print(f"Document: {metadata.get('document_id')}")
            print(f"Chunk: {metadata.get('chunk_index')}")
            print(f"L2 Distance: {score:.4f}")
            print(f"Content: {document.page_content[:400]}")
            print("-" * 60)


if __name__ == "__main__":
    main()
