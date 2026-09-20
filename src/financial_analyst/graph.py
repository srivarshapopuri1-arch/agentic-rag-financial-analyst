from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .agents import build_llm, format_context, validate_answer
from .config import Settings
from .tools import calculator
from .vectorstore import retrieve


class AnalystState(TypedDict, total=False):
    question: str
    evidence: list[Document]
    messages: Annotated[list[BaseMessage], add_messages]
    answer: str
    grounded: bool
    validation_message: str


SYSTEM_PROMPT = """You are a financial-document research and analysis agent.

Use ONLY the supplied evidence. Do not use memory or outside facts.

Do not invent figures, periods, companies, causes, KPIs, or conclusions.

For every material factual claim, cite the supporting marker exactly as supplied.

For comparisons, identify each company or period and do not compare
incompatible units.

When arithmetic is required, use the calculator tool instead of doing the
calculation yourself.

After the calculator returns a result, produce a complete final answer that:
1. states the calculation inputs and formula,
2. states the calculator result,
3. cites the supplied financial-document evidence using the exact citation
   marker provided with that evidence.

Calculator output is not documentary evidence. Numerical inputs taken from
the financial documents must still be cited to their source.

If evidence is missing, ambiguous, or conflicting, say so clearly.

This is document analysis, not investment advice.
"""


def build_graph(store, settings: Settings):
    """Build the agentic financial-document analysis workflow."""
    llm = build_llm(settings)
    llm_with_tools = llm.bind_tools([calculator])

    def document_research(state: AnalystState) -> AnalystState:
        evidence = retrieve(
            store,
            state["question"],
            settings.top_k,
        )

        if not evidence:
            return {
                "evidence": [],
                "messages": [
                    AIMessage(
                        content=(
                            "Insufficient evidence in the indexed documents "
                            "to answer this question."
                        )
                    )
                ],
            }

        prompt = (
            f"Question:\n{state['question']}\n\n"
            f"Evidence:\n{format_context(evidence)}"
        )

        return {
            "evidence": evidence,
            "messages": [HumanMessage(content=prompt)],
        }

    def financial_analysis(state: AnalystState) -> AnalystState:
        evidence = state.get("evidence", [])

        if not evidence:
            return {}

        response = llm_with_tools.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                *state.get("messages", []),
            ]
        )

        return {"messages": [response]}

    def route_after_analysis(state: AnalystState) -> str:
        messages = state.get("messages", [])

        if not messages:
            return "finalize_answer"

        last_message = messages[-1]

        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "calculation_tools"

        return "finalize_answer"

    def finalize_answer(state: AnalystState) -> AnalystState:
        evidence = state.get("evidence", [])

        if not evidence:
            return {
                "answer": (
                    "Insufficient evidence in the indexed documents "
                    "to answer this question."
                )
            }

        final_prompt = f"""Produce the final answer to the user's financial question.

Original question:
{state["question"]}

Retrieved financial evidence:
{format_context(evidence)}

The conversation below may contain calculator tool calls and results.

Requirements:
- Use only the retrieved financial evidence for factual claims.
- Preserve calculator results when a calculator tool was used.
- State the calculation inputs and formula when arithmetic was performed.
- Cite every material financial claim using the EXACT citation markers shown
  in the retrieved evidence, for example [filename.pdf p. 10].
- Do not cite the calculator as a source.
- Do not invent missing information.
- If the evidence does not support the requested conclusion, explicitly say
  "Insufficient evidence."

Return only the final answer.
"""

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You produce grounded final answers from retrieved "
                        "financial-document evidence."
                    )
                ),
                *state.get("messages", []),
                HumanMessage(content=final_prompt),
            ]
        )

        return {"answer": str(response.content)}

    def response_validation(state: AnalystState) -> AnalystState:
        grounded, message = validate_answer(
            state.get("answer", ""),
            state.get("evidence", []),
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

        return {
            "grounded": True,
            "validation_message": message,
        }

    graph = StateGraph(AnalystState)

    graph.add_node("document_research", document_research)
    graph.add_node("financial_analysis", financial_analysis)
    graph.add_node("calculation_tools", ToolNode([calculator]))
    graph.add_node("finalize_answer", finalize_answer)
    graph.add_node("response_validation", response_validation)

    graph.add_edge(START, "document_research")
    graph.add_edge("document_research", "financial_analysis")

    graph.add_conditional_edges(
        "financial_analysis",
        route_after_analysis,
        {
            "calculation_tools": "calculation_tools",
            "finalize_answer": "finalize_answer",
        },
    )

    graph.add_edge("calculation_tools", "financial_analysis")
    graph.add_edge("finalize_answer", "response_validation")
    graph.add_edge("response_validation", END)

    return graph.compile()