
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    """
    Split LangChain Documents into smaller chunks.

    Each chunk retains the original metadata and receives:
    - chunk_index: position within its source document
    - start_index: character position in the original content
    """

    if not isinstance(documents, list):
        raise TypeError("documents must be a list")

    if not all(isinstance(doc, Document) for doc in documents):
        raise TypeError("Every item must be a LangChain Document")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    # Assign a sequential index within each source document.
    chunk_counts = {}

    for chunk in chunks:
        source_id = chunk.metadata.get(
            "document_id",
            chunk.metadata.get("source", "unknown"),
        )

        chunk_index = chunk_counts.get(source_id, 0)

        chunk.metadata["chunk_index"] = chunk_index

        chunk_counts[source_id] = chunk_index + 1

    return chunks
