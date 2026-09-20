from langchain_core.documents import Document

from financial_analyst.agents import validate_answer


def evidence():
    return [
        Document(
            page_content="Revenue was reported.",
            metadata={"source": "annual-report.pdf", "page": 42},
        )
    ]


def test_valid_retrieved_citation_passes():
    grounded, _ = validate_answer(
        "Revenue is discussed in the filing [annual-report.pdf p. 42].",
        evidence(),
    )
    assert grounded is True


def test_unknown_citation_fails():
    grounded, message = validate_answer(
        "Revenue increased [other.pdf p. 9].",
        evidence(),
    )
    assert grounded is False
    assert "not retrieved" in message


def test_no_evidence_requires_insufficient_evidence():
    grounded, _ = validate_answer(
        "Insufficient evidence in the indexed documents to answer this question.",
        [],
    )
    assert grounded is True
