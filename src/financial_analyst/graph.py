from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from .agents import analyze_question, build_llm, validate_answer
from .config import Settings
from .vectorstore import retrieve


class AnalystState(TypedDict, total=False):
    question: str
    evidence: list[Document]
    answer: str
    grounded: bool
    validation_message: str


def build_graph(store, settings: Settings):
    """Build the document research -> financial analysis -> validation workflow."""
    llm = build_llm(settings)

    def document_research(state: AnalystState) -> AnalystState:
        return {"evidence": retrieve(store, state["question"], settings.top_k)}

    def financial_analysis(state: AnalystState) -> AnalystState:
        return {
            "answer": analyze_question(
                llm, state["question"], state.get("evidence", [])
            )
        }

    def response_validation(state: AnalystState) -> AnalystState:
        grounded, message = validate_answer(
            state.get("answer", ""), state.get("evidence", [])
        )
        if not grounded:
            return {
                "grounded": False,
                "validation_message": message,
                "answer": (
                    "Insufficient evidence to provide a grounded answer. "
                    f"Validation detail: {message}"
                ),
            }
        return {"grounded": True, "validation_message": message}

    graph = StateGraph(AnalystState)
    graph.add_node("document_research", document_research)
    graph.add_node("financial_analysis", financial_analysis)
    graph.add_node("response_validation", response_validation)
    graph.add_edge(START, "document_research")
    graph.add_edge("document_research", "financial_analysis")
    graph.add_edge("financial_analysis", "response_validation")
    graph.add_edge("response_validation", END)
    return graph.compile()
