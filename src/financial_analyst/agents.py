from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .config import Settings

_CITATION_PATTERN = re.compile(r"\[([^\[\]]+?\s+p\.\s*\d+)\]")


def citation_for(document: Document) -> str:
    return f"[{document.metadata.get('source', 'unknown')} p. {document.metadata.get('page', '?')}]"


def format_context(documents: list[Document]) -> str:
    return "\n\n".join(f"{citation_for(doc)}\n{doc.page_content}" for doc in documents)


def build_llm(settings: Settings) -> ChatOpenAI:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for LLM analysis.")
    return ChatOpenAI(model=settings.llm_model, api_key=settings.openai_api_key, temperature=0)


def analyze_question(llm: ChatOpenAI, question: str, evidence: list[Document]) -> str:
    """Generate an answer constrained to retrieved financial evidence."""
    if not evidence:
        return "Insufficient evidence in the indexed documents to answer this question."

    system = """You are a financial-document research and analysis agent.
Use ONLY the supplied evidence. Do not use memory or outside facts.
Do not invent figures, periods, companies, causes, KPIs, or conclusions.
For every material factual claim, cite the supporting marker exactly as supplied.
For comparisons, identify each company/period and do not compare incompatible units.
When arithmetic is useful, state the inputs and formula so the result can be verified.
If evidence is missing, ambiguous, or conflicting, say so clearly.
This is document analysis, not investment advice."""
    prompt = f"Question:\n{question}\n\nEvidence:\n{format_context(evidence)}"
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
    return str(response.content)


def validate_answer(answer: str, evidence: list[Document]) -> tuple[bool, str]:
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
        return False, f"Answer referenced evidence that was not retrieved: {sorted(unknown)}"
    return True, "Citations refer only to retrieved evidence."
