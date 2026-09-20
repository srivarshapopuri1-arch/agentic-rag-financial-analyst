import pytest
from langchain_core.documents import Document

from financial_analyst.ingestion import chunk_documents


def test_chunking_preserves_metadata():
    document = Document(
        page_content="Revenue increased. " * 100,
        metadata={"source": "sample.pdf", "page": 7},
    )
    chunks = chunk_documents([document], chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "sample.pdf" for chunk in chunks)
    assert all(chunk.metadata["page"] == 7 for chunk in chunks)
    assert [chunk.metadata["chunk_id"] for chunk in chunks] == list(range(len(chunks)))


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValueError):
        chunk_documents([], chunk_size=200, chunk_overlap=200)
