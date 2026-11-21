
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


def generate_embeddings(
    chunks: list[Document],
    embedding_model=None,
    model_name: str = "text-embedding-3-small",
) -> list[list[float]]:
    """
    Generate numerical embeddings for document chunks.

    Args:
        chunks: List of LangChain Document chunks.
        embedding_model: Optional embedding model for dependency injection.
        model_name: OpenAI embedding model name.

    Returns:
        A list of numerical vectors, one per chunk.
    """

    if not isinstance(chunks, list):
        raise TypeError("chunks must be a list")

    if not all(isinstance(chunk, Document) for chunk in chunks):
        raise TypeError("Every item must be a LangChain Document")

    if any(not chunk.page_content.strip() for chunk in chunks):
        raise ValueError("Chunks must not contain empty text")

    if not chunks:
        return []

    if embedding_model is None:
        load_dotenv()
        embedding_model = OpenAIEmbeddings(model=model_name)

    texts = [chunk.page_content for chunk in chunks]

    vectors = embedding_model.embed_documents(texts)

    if len(vectors) != len(chunks):
        raise ValueError(
            "Number of embeddings does not match number of chunks"
        )

    if not vectors:
        raise ValueError("Embedding model returned no vectors")

    dimension = len(vectors[0])

    if dimension == 0:
        raise ValueError("Embedding vectors must not be empty")

    for vector in vectors:
        if len(vector) != dimension:
            raise ValueError("Embedding dimensions are inconsistent")

        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            for value in vector
        ):
            raise ValueError("Embedding vectors must contain numbers")

    return vectors
