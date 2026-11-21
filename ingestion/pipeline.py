
from pathlib import Path

from langchain_core.documents import Document

from ingestion.document_loader import load_markdown_documents
from ingestion.document_cleaner import clean_documents
from ingestion.chunking import split_documents
from ingestion.vector_store import build_vector_store, save_vector_store


def run_ingestion_pipeline(
    documents_dir: str,
    vector_store_dir: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    embedding_model=None,
) -> dict:
    """
    Execute the complete document ingestion pipeline.

    Steps:
        1. Load Markdown documents.
        2. Clean document content.
        3. Split documents into chunks.
        4. Generate embeddings and build FAISS index.
        5. Persist the FAISS vector store.

    embedding_model is injectable for testing.
    """

    # Step 1: Load documents
    documents = load_markdown_documents(documents_dir)

    # Step 2: Clean documents
    cleaned_documents = clean_documents(documents)

    # Step 3: Split documents into chunks
    chunks = split_documents(
        cleaned_documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not chunks:
        raise ValueError("No chunks were generated from the documents")

    # Step 4: Build FAISS index.
    # FAISS generates embeddings internally.
    vector_store = build_vector_store(
        chunks,
        embedding_model=embedding_model,
    )

    # Step 5: Persist the vector store
    save_vector_store(
        vector_store,
        vector_store_dir,
    )

    return {
        "documents_loaded": len(documents),
        "documents_cleaned": len(cleaned_documents),
        "chunks_created": len(chunks),
        "vectors_stored": vector_store.index.ntotal,
        "vector_store_path": str(Path(vector_store_dir)),
        "vector_store": vector_store,
    }
