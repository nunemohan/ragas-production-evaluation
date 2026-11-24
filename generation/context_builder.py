
from langchain_core.documents import Document


def build_context(documents: list[Document]) -> str:
    """
    Convert retrieved documents into a structured context string.

    Each chunk includes its source document and chunk index
    so that the LLM can understand where the information came from.
    """
    if not isinstance(documents, list):
        raise ValueError("Documents must be provided as a list")

    if not documents:
        return "No relevant context was retrieved."

    context_parts = []

    for index, document in enumerate(documents, start=1):
        if not isinstance(document, Document):
            raise ValueError(
                f"Item at position {index} must be a LangChain Document"
            )

        content = document.page_content.strip()

        if not content:
            continue

        # Add this line here
        index = len(context_parts) + 1

        metadata = document.metadata or {}

        source = metadata.get("source_filename") or metadata.get(
            "document_id", "Unknown source"
        )

        chunk_index = metadata.get("chunk_index", "Unknown")

        context_part = (
            f"[Context {index}]\n"
            f"Source: {source}\n"
            f"Chunk: {chunk_index}\n"
            f"Content:\n{content}"
        )

        context_parts.append(context_part)

    if not context_parts:
        return "No relevant context was retrieved."

    return "\n\n---\n\n".join(context_parts)
