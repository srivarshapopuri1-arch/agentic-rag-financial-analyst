from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from .config import Settings

_CITATION_PATTERN = re.compile(r"\[([^\[\]]+?\s+p\.\s*\d+)\]")


def citation_for(document: Document) -> str:
    """Return the citation marker associated with a retrieved document."""
    return (
        f"[{document.metadata.get('source', 'unknown')} "
        f"p. {document.metadata.get('page', '?')}]"
    )


def format_context(documents: list[Document]) -> str:
    """Format retrieved documents as citation-labelled evidence."""
    return "\n\n".join(
        f"{citation_for(doc)}\n{doc.page_content}" for doc in documents
    )


def build_llm(settings: Settings) -> ChatOllama:
    """Create the local Ollama chat model."""
    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )


def validate_answer(
    answer: str,
    evidence: list[Document],
) -> tuple[bool, str]:
    """Apply deterministic evidence/citation checks to a generated answer."""
    insufficient = "insufficient evidence" in answer.lower()

    if not evidence:
        return insufficient, "No retrieved evidence was available."

    allowed = {citation_for(doc).strip("[]") for doc in evidence}
    citations = set(_CITATION_PATTERN.findall(answer))

    if insufficient and not citations:
        return True, "The answer explicitly reports insufficient evidence."

    if not citations:
        return False, "The answer contains no source citations."

    unknown = citations - allowed

    if unknown:
        return (
            False,
            f"Answer referenced evidence that was not retrieved: {sorted(unknown)}",
        )

    return True, "Citations refer only to retrieved evidence."
