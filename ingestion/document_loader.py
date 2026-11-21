
from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
)
from langchain_core.documents import Document


def load_markdown_documents(
    documents_dir: str,
) -> list[Document]:
    """
    Load all Markdown documents from the given directory.

    Args:
        documents_dir: Directory containing Markdown files.

    Returns:
        List of LangChain Document objects.
    """

    directory = Path(documents_dir)

    if not directory.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {directory}"
        )

    if not directory.is_dir():
        raise NotADirectoryError(
            f"Expected a directory: {directory}"
        )

    markdown_files = list(directory.glob("*.md"))

    if not markdown_files:
        raise ValueError(
            f"No Markdown documents found in: {directory}"
        )

    loader = DirectoryLoader(
        path=str(directory),
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
        use_multithreading=False,
    )

    documents = loader.load()

    # Maintain deterministic document ordering.
    documents.sort(
        key=lambda document: document.metadata["source"]
    )

    # Enrich metadata for source traceability.
    for document in documents:
        source_path = Path(document.metadata["source"])

        document.metadata["source_filename"] = (
            source_path.name
        )

        document.metadata["document_id"] = (
            source_path.stem
        )

    return documents
