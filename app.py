import streamlit as st

from financial_analyst.config import get_settings
from financial_analyst.graph import build_graph
from financial_analyst.ingestion import load_directory
from financial_analyst.logging_config import configure_logging
from financial_analyst.vectorstore import build_vector_store, index_documents

settings = get_settings()
configure_logging(settings.log_level)

st.set_page_config(page_title="Agentic RAG Financial Analyst", page_icon="📊", layout="wide")
st.title("Agentic RAG Financial Analyst")
st.caption("Document-grounded research over public financial reports. Not investment advice.")

if not settings.openai_api_key:
    st.warning("Set OPENAI_API_KEY in your local .env file before indexing or asking questions.")

with st.sidebar:
    st.header("Document index")
    st.write(f"PDF directory: `{settings.data_dir}`")
    if st.button("Index documents", use_container_width=True):
        try:
            with st.spinner("Parsing, chunking, embedding, and indexing PDFs..."):
                documents = load_directory(
                    settings.data_dir, settings.chunk_size, settings.chunk_overlap
                )
                store = build_vector_store(settings)
                count = index_documents(store, documents)
            st.success(f"Indexed {count} chunks.")
        except Exception as exc:
            st.error(str(exc))

question = st.text_area(
    "Ask a financial question",
    placeholder="Example: What revenue was reported for the latest fiscal year? Cite the source.",
)

if st.button("Analyze", type="primary", disabled=not bool(question.strip())):
    try:
        with st.spinner("Researching documents and validating the answer..."):
            store = build_vector_store(settings)
            graph = build_graph(store, settings)
            result = graph.invoke({"question": question.strip()})
        st.subheader("Analysis")
        st.write(result["answer"])
        if result.get("grounded"):
            st.success("Grounding check passed.")
        else:
            st.warning(result.get("validation_message", "Grounding could not be verified."))

        with st.expander("Retrieved evidence"):
            for doc in result.get("evidence", []):
                source = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page", "?")
                st.markdown(f"**{source} — page {page}**")
                st.write(doc.page_content[:1200])
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")

st.divider()
st.caption("Use public or appropriately licensed documents only. Verify material conclusions in the original filing.")
