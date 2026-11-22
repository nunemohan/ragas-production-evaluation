
from pathlib import Path

from ingestion.vector_store import (
    load_vector_store,
    search_similar_documents,
)


def create_retriever(
    vector_store_dir: str,
    embedding_model=None,
):
    """
    Load the persisted FAISS vector store and return it.
    """
    if not Path(vector_store_dir).exists():
        raise FileNotFoundError(
            f"Vector store directory not found: {vector_store_dir}"
        )

    return load_vector_store(
        directory=vector_store_dir,
        embedding_model=embedding_model,
    )


def retrieve_documents(
    vector_store,
    query: str,
    top_k: int = 3,
):
    """
    Retrieve the top-K relevant document chunks.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string")

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    return search_similar_documents(
        vector_store=vector_store,
        query=query.strip(),
        top_k=top_k,
    )
