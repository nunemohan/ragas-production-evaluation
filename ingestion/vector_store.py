
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


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
        load_dotenv()
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
        load_dotenv()
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

    return FAISS.load_local(
        str(path),
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )



def search_similar_documents(
    vector_store,
    query: str,
    top_k: int = 3,
    metadata_filter: dict | None = None,
):
    """
    Search FAISS for relevant documents,
    optionally restricting results using metadata.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string")

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    if metadata_filter is not None and not isinstance(metadata_filter, dict):
        raise ValueError("metadata_filter must be a dictionary or None")

    return vector_store.similarity_search(
        query=query.strip(),
        k=top_k,
        filter=metadata_filter,
    )

