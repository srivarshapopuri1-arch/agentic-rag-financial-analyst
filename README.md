# Agentic RAG Financial Analyst

A portfolio-scale financial research assistant that combines retrieval-augmented generation (RAG), LangGraph orchestration, financial analysis, safe calculations, citations, and grounding validation.

## Features

- Ingests public 10-K, 10-Q, annual-report, and other financial PDFs.
- Extracts page-level text with PyMuPDF and preserves source/page metadata.
- Chunks and embeds documents into a persistent Chroma vector database.
- Retrieves evidence for financial questions, KPI extraction, and company/period comparisons.
- Orchestrates document research, financial analysis, and response validation with LangGraph.
- Includes a restricted arithmetic calculator for verifiable financial calculations.
- Requires citations to retrieved evidence and rejects unknown citations.
- Returns an explicit insufficient-evidence response instead of knowingly presenting an ungrounded answer.
- Provides a Streamlit interface, configuration management, logging, and offline unit tests.

> This project is for research and portfolio demonstration only. It is not investment advice.

## Architecture

```text
Public financial PDFs
        |
        v
PyMuPDF ingestion
        |
        v
Chunking + source/page metadata
        |
        v
OpenAI embeddings -> Chroma
        |
        v
User question
        |
        v
LangGraph
  document research
        |
  financial analysis
        |
  response validation
        |
        v
Cited answer / insufficient evidence
        |
        v
Streamlit UI
```

The workflow is intentionally compact enough for an individual portfolio project. Retrieval establishes the evidence set, the analysis agent is constrained to that evidence, and a deterministic validation step checks whether citations refer to retrieved sources.

## Repository structure

```text
.
├── app.py
├── pyproject.toml
├── .env.example
├── .gitignore
├── data/
│   └── README.md
├── src/
│   └── financial_analyst/
│       ├── __init__.py
│       ├── agents.py
│       ├── config.py
│       ├── graph.py
│       ├── ingestion.py
│       ├── logging_config.py
│       ├── tools.py
│       └── vectorstore.py
└── tests/
    ├── test_ingestion.py
    ├── test_tools.py
    └── test_validation.py
```

## Setup

Python 3.11+ is recommended.

```bash
git clone https://github.com/srivarshapopuri1-arch/agentic-rag-financial-analyst.git
cd agentic-rag-financial-analyst
python -m venv .venv
```

Activate the environment and install the package:

```bash
pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and add your own API key. Never commit `.env`.

macOS/Linux:

```bash
cp .env.example .env
```

Windows Command Prompt:

```cmd
copy .env.example .env
```

## Financial documents

Place legitimate public financial-report PDFs in `data/`. PDFs are deliberately ignored by Git. Appropriate inputs include public annual reports or regulatory filing PDFs that you are permitted to use.

Do not add confidential employer/customer documents, credentials, or private financial information.

## Run

```bash
streamlit run app.py
```

Use **Index documents** in the sidebar, then ask questions.

Example queries:

- What revenue did the company report for the latest fiscal year?
- Compare revenue across the two periods represented in these filings.
- What liquidity risks does management describe?
- Extract operating income and explain the reported year-over-year change.
- Which retrieved evidence supports the company's reported capital expenditures?

For arithmetic, the project also contains a restricted calculator tool. It accepts numeric arithmetic but rejects Python calls, names, imports, and attribute access.

## Workflow and grounding

1. **Document research** retrieves the top relevant chunks from Chroma.
2. **Financial analysis** asks the LLM to use only those chunks and attach the supplied source/page markers to material claims.
3. **Response validation** checks that citations exist and that every cited marker belongs to the retrieved evidence set.
4. When evidence is absent or grounding fails, the application returns an insufficient-evidence response rather than silently inventing support.

This improves grounding but does not prove that every interpretation is economically correct. Material conclusions should always be checked against the original filing.

## Configuration

`.env.example` documents the available settings:

- `OPENAI_API_KEY`
- `LLM_MODEL`
- `EMBEDDING_MODEL`
- `CHROMA_DIR`
- `COLLECTION_NAME`
- `DATA_DIR`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `TOP_K`
- `LOG_LEVEL`

## Tests

Run:

```bash
pytest
ruff check .
```

The tests are intentionally offline and cover chunk metadata, chunk configuration validation, arithmetic, rejection of code execution, citation validation, and no-evidence behavior. Live OpenAI/embedding integration requires a valid API key and is not represented as an offline unit test.

## Limitations

- Image-only/scanned PDFs require OCR, which is not implemented.
- Complex tables can lose structure during plain PDF text extraction.
- Vector retrieval can miss relevant passages.
- Citation validation checks source-marker grounding; it is not a full factual-entailment evaluator.
- Cross-company and cross-period comparisons are only as complete as the indexed filings and retrieved evidence.
- The LLM and embedding workflow requires a user-supplied API key and network access.
- This is a portfolio research assistant, not an investment-advice or production financial-data platform.

## Possible extensions

Useful future work includes SEC EDGAR API ingestion, table-aware parsing, reranking, structured KPI extraction schemas, curated RAG evaluation datasets, tracing, and a FastAPI service layer.

## Responsible use

Use public or appropriately licensed documents only. Never commit API keys, credentials, private data, or confidential financial information.
