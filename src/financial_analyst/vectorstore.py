from __future__ import annotations

import hashlib

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from .config import Settings


def build_vector_store(settings: Settings) -> Chroma:
    """Create a persistent Chroma store backed by OpenAI embeddings."""
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for embeddings.")

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )


def index_documents(store: Chroma, documents: list[Document]) -> int:
    """Index chunks with deterministic IDs to avoid duplicate copies on re-index."""
    if not documents:
        return 0

    ids: list[str] = []
    for document in documents:
        raw = (
            f"{document.metadata.get('source')}|{document.metadata.get('page')}|"
            f"{document.metadata.get('chunk_id')}|{document.page_content}"
        )
        ids.append(hashlib.sha256(raw.encode("utf-8")).hexdigest())

    store.add_documents(documents=documents, ids=ids)
    return len(documents)


def retrieve(store: Chroma, query: str, top_k: int = 6) -> list[Document]:
    return store.similarity_search(query, k=top_k)
