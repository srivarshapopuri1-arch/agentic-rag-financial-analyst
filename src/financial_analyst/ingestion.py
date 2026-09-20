from __future__ import annotations

from pathlib import Path

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(path: Path) -> list[Document]:
    """Extract non-empty PDF pages while retaining source/page metadata."""
    if not path.exists() or path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected an existing PDF file: {path}")

    documents: list[Document] = []
    with pymupdf.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": path.name, "page": page_number},
                    )
                )
    return documents


def chunk_documents(
    documents: list[Document], chunk_size: int = 1200, chunk_overlap: int = 200
) -> list[Document]:
    """Split pages into retrieval chunks without losing source metadata."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index
    return chunks


def load_directory(
    directory: Path, chunk_size: int = 1200, chunk_overlap: int = 200
) -> list[Document]:
    """Load and chunk every PDF in a directory."""
    if not directory.exists():
        raise FileNotFoundError(f"Data directory does not exist: {directory}")

    pdfs = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {directory}")

    pages: list[Document] = []
    for pdf in pdfs:
        pages.extend(load_pdf(pdf))
    return chunk_documents(pages, chunk_size, chunk_overlap)
