
"""
Citation builder for the RAG application.

Extracts source references from retrieved LangChain Documents.
"""

from pathlib import Path

from langchain_core.documents import Document


def build_citations(documents: list[Document]) -> list[dict]:
    """
    Build unique source citations from retrieved documents.

    Each citation contains:
        document_id: Source document identifier.
        source: Original Markdown filename.
        chunk_index: Retrieved chunk position, when available.
    """

    if not isinstance(documents, list):
        raise ValueError("Documents must be a list")

    citations = []
    seen = set()

    for document in documents:
        if not isinstance(document, Document):
            raise ValueError(
                "Every item must be a LangChain Document"
            )

        metadata = document.metadata

        source = metadata.get("source_filename")

        if not source:
            original_source = metadata.get("source")
            if original_source:
                source = Path(original_source).name

        document_id = metadata.get("document_id")

        if not document_id and source:
            document_id = Path(source).stem

        if not source and not document_id:
            continue

        chunk_index = metadata.get("chunk_index")

        citation_key = (document_id, source, chunk_index)

        if citation_key in seen:
            continue

        seen.add(citation_key)

        citation = {
            "document_id": document_id,
            "source": source,
        }

        if chunk_index is not None:
            citation["chunk_index"] = chunk_index

        citations.append(citation)

    return citations
