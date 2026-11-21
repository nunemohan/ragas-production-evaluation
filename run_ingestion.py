
from pathlib import Path

from ingestion.pipeline import run_ingestion_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"


def main():
    print("\nStarting RAG document ingestion pipeline...\n")

    result = run_ingestion_pipeline(
        documents_dir=str(DOCUMENTS_DIR),
        vector_store_dir=str(VECTOR_STORE_DIR),
        chunk_size=500,
        chunk_overlap=50,
    )

    print("\nIngestion completed successfully!")
    print("-----------------------------------")
    print(f"Documents loaded  : {result['documents_loaded']}")
    print(f"Documents cleaned : {result['documents_cleaned']}")
    print(f"Chunks created    : {result['chunks_created']}")
    print(f"Vectors stored    : {result['vectors_stored']}")
    print(f"Vector store path : {result['vector_store_path']}")


if __name__ == "__main__":
    main()
