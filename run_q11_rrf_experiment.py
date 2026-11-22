
from pathlib import Path

from retrieval.retriever import create_retriever
from retrieval.rrf import reciprocal_rank_fusion


PROJECT_ROOT = Path(__file__).resolve().parent
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"

QUERIES = {
    "Expense windows": (
        "How many days before admission and after discharge "
        "are eligible hospitalization expenses covered?"
    ),
    "Reimbursement documents": (
        "What documents must be submitted for a reimbursement "
        "claim after treatment at a non-network hospital?"
    ),
}


def main():
    vector_store = create_retriever(str(VECTOR_STORE_DIR))

    result_lists = []

    for query_name, query in QUERIES.items():
        print(f"\nSearching: {query_name}")

        results = vector_store.similarity_search(
            query=query,
            k=5,
        )

        result_lists.append(results)

        for rank, document in enumerate(results, start=1):
            print(
                f"Rank {rank}: "
                f"{document.metadata.get('document_id')} | "
                f"Chunk {document.metadata.get('chunk_index')}"
            )

    fused_results = reciprocal_rank_fusion(
        result_lists=result_lists,
        k=60,
        top_k=10,
    )

    print("\n" + "=" * 70)
    print("RRF FUSED RESULTS")
    print("=" * 70)

    for rank, (document, score) in enumerate(fused_results, start=1):
        print(f"\nRank: {rank}")
        print(f"Document: {document.metadata.get('document_id')}")
        print(f"Chunk: {document.metadata.get('chunk_index')}")
        print(f"RRF Score: {score:.6f}")
        print(f"Content: {document.page_content[:400]}")


if __name__ == "__main__":
    main()
