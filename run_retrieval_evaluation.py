
from pathlib import Path

from retrieval.retriever import create_retriever
from evaluation.evaluator import evaluate_golden_dataset


PROJECT_ROOT = Path(__file__).resolve().parent

VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"
GOLDEN_DATASET = (
    PROJECT_ROOT
    / "data"
    / "golden_dataset"
    / "ragas_golden_dataset.json"
)


def main():
    print("\nStarting retrieval quality evaluation...\n")

    vector_store = create_retriever(str(VECTOR_STORE_DIR))

    results = evaluate_golden_dataset(
        dataset_path=str(GOLDEN_DATASET),
        vector_store=vector_store,
        k_values=(1, 3, 5),
    )

    print("Dataset:", results["dataset_name"])
    print("Total questions:", results["total_questions"])
    print("Answerable:", results["answerable_questions"])
    print("Unanswerable:", results["unanswerable_questions"])

    print("\nRetrieval Metrics")
    print("-----------------")

    for k, metrics in results["summary"].items():
        print(f"\n{k}")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")



    print("\nQuestion-level Evaluation")
    print("=========================")

    for item in results["question_results"]:
        print(f"\nQuestion: {item['question_id']}")
        print(f"Category: {item['category']}")
        print(f"Query: {item['user_input']}")

        if item["is_unanswerable"]:
            print("Type: Unanswerable")
            print("Retrieved documents:")

            for k, metrics in item["metrics"].items():
                print(
                    f"{k}: {metrics['retrieved_document_ids']}"
                )
            continue

        for k, metrics in item["metrics"].items():
            print(
                f"{k}: "
                f"Precision={metrics['precision']:.2f}, "
                f"Recall={metrics['recall']:.2f}, "
                f"Hit Rate={metrics['hit_rate']}"
            )
            print(
                "Retrieved:",
                metrics["retrieved_document_ids"]
            )



if __name__ == "__main__":
    main()
