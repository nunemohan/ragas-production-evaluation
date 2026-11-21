
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def build_vector_store(
    chunks: list[Document],
    embedding_model=None,
) -> FAISS:
    """
    Build a FAISS vector store from document chunks.

    FAISS uses the embedding model to generate vectors
    and stores the corresponding documents and metadata.
    """

    if not isinstance(chunks, list):
        raise TypeError("chunks must be a list")

    if not chunks:
        raise ValueError("At least one document chunk is required")

    if not all(isinstance(chunk, Document) for chunk in chunks):
        raise TypeError("Every item must be a LangChain Document")

    if any(not chunk.page_content.strip() for chunk in chunks):
        raise ValueError("Chunks must not contain empty text")

    if embedding_model is None:
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

    return FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model,
    )


def save_vector_store(
    vector_store: FAISS,
    directory: str,
) -> None:
    """Save a FAISS vector store to a local directory."""

    if not isinstance(vector_store, FAISS):
        raise TypeError("vector_store must be a FAISS instance")

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

    vector_store.save_local(str(path))


def load_vector_store(
    directory: str,
    embedding_model=None,
) -> FAISS:
    """
    Load a previously saved FAISS vector store.

    Only load indexes from trusted sources because LangChain
    uses pickle-based metadata storage.
    """

    path = Path(directory)

    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(
            f"Vector store directory not found: {path}"
        )

    index_file = path / "index.faiss"
    metadata_file = path / "index.pkl"

    if not index_file.exists() or not metadata_file.exists():
        raise FileNotFoundError(
            f"FAISS index files are missing in: {path}"
        )

    if embedding_model is None:
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

    return FAISS.load_local(
        str(path),
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )


def search_similar_documents(
    vector_store: FAISS,
    query: str,
    top_k: int = 3,
) -> list[Document]:
    """Retrieve the top-K most relevant document chunks."""

    if not isinstance(vector_store, FAISS):
        raise TypeError("vector_store must be a FAISS instance")

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    return vector_store.similarity_search(
        query=query,
        k=top_k,
    )
