
from pathlib import Path

from retrieval.retriever import create_retriever
from retrieval.rrf import reciprocal_rank_fusion
from retrieval.reranker import rerank_documents_for_queries


PROJECT_ROOT = Path(__file__).resolve().parent
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"

QUERY_A = (
    "How many days before admission and after discharge "
    "are eligible hospitalization expenses covered?"
)

QUERY_B = (
    "What documents must be submitted for a reimbursement "
    "claim after treatment at a non-network hospital?"
)


def main():
    vector_store = create_retriever(str(VECTOR_STORE_DIR))

    # Step 1: FAISS retrieval
    results_a = vector_store.similarity_search(QUERY_A, k=5)
    results_b = vector_store.similarity_search(QUERY_B, k=5)

    # Step 2: RRF fusion
    fused_results = reciprocal_rank_fusion(
        [results_a, results_b],
        k=60,
        top_k=10,
    )

    print("\n========== RRF RESULTS ==========")

    for rank, (document, score) in enumerate(fused_results, 1):
        print(
            f"Rank {rank}: "
            f"{document.metadata.get('document_id')} | "
            f"Chunk {document.metadata.get('chunk_index')} | "
            f"Score: {score:.6f}"
        )

    candidate_documents = [
        document for document, _ in fused_results
    ]

    # Step 3: Subquery-aware Cross-Encoder reranking
    reranked_results = rerank_documents_for_queries(
        queries=[QUERY_A, QUERY_B],
        documents=candidate_documents,
        top_k=8,
    )

    # Step 4: Display independent rankings
    for query, results in reranked_results.items():
        print(f"\n========== CROSS-ENCODER: {query} ==========")

        for rank, (document, score) in enumerate(results, 1):
            print(
                f"Rank {rank}: "
                f"{document.metadata.get('document_id')} | "
                f"Chunk {document.metadata.get('chunk_index')} | "
                f"Score: {score:.6f}"
            )


if __name__ == "__main__":
    main()
