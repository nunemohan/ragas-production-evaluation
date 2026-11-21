
import re
from langchain_core.documents import Document


def clean_document(document: Document) -> Document:
    """
    Clean a LangChain Document while preserving
    its original metadata and meaningful content.

    Args:
        document: Input LangChain Document.

    Returns:
        A new cleaned LangChain Document.
    """

    if not isinstance(document, Document):
        raise TypeError("Input must be a LangChain Document")

    content = document.page_content

    # 1. Normalize Windows and Mac line endings.
    content = content.replace("\r\n", "\n")
    content = content.replace("\r", "\n")

    # 2. Remove null characters.
    content = content.replace("\x00", "")

    # 3. Normalize non-breaking spaces.
    content = content.replace("\u00a0", " ")

    # 4. Remove trailing whitespace from each line.
    lines = [
        line.rstrip()
        for line in content.split("\n")
    ]

    content = "\n".join(lines)

    # 5. Collapse three or more consecutive newlines
    #    into two newlines.
    content = re.sub(r"\n{3,}", "\n\n", content)

    # 6. Remove unnecessary leading and trailing whitespace.
    content = content.strip()

    # 7. Return a new Document and preserve metadata.
    return Document(
        page_content=content,
        metadata=document.metadata.copy()
    )


def clean_documents(
    documents: list[Document]
) -> list[Document]:
    """
    Clean a collection of LangChain Documents.

    Args:
        documents: List of LangChain Documents.

    Returns:
        List of cleaned Documents.
    """

    return [
        clean_document(document)
        for document in documents
    ]
