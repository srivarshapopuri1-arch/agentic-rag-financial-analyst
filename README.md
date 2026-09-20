# Agentic RAG Financial Analyst

A local-first financial research assistant that combines retrieval-augmented generation (RAG), LangGraph orchestration, tool calling, financial analysis, source citations, and grounding validation.

The application analyzes public financial-report PDFs using local models through Ollama, allowing the core workflow to run without paid LLM API credits.

## Features

- Ingests public 10-K, 10-Q, annual-report, and other financial PDFs.
- Extracts page-level text with PyMuPDF while preserving source and page metadata.
- Chunks documents and generates local embeddings with `nomic-embed-text`.
- Stores and retrieves financial evidence using persistent Chroma vector search.
- Uses `llama3.2:3b` locally through Ollama for financial-document analysis.
- Orchestrates retrieval, analysis, calculator tool calling, final-answer generation, and validation with LangGraph.
- Includes a restricted arithmetic calculator for verifiable financial calculations.
- Requires source/page citations for grounded financial claims.
- Rejects citations that do not belong to the retrieved evidence set.
- Returns an explicit insufficient-evidence response when a grounded answer cannot be produced.
- Provides a Streamlit interface, configuration management, logging, and offline unit tests.
- Runs locally without requiring an OpenAI API key.

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
nomic-embed-text (Ollama)
        |
        v
Persistent Chroma vector store
        |
        v
User financial question
        |
        v
LangGraph workflow
        |
        +--> Document research
        |
        +--> Financial analysis (Llama 3.2)
        |
        +--> Calculator tool when arithmetic is required
        |          |
        |          +----> Financial analysis
        |
        +--> Final grounded answer
        |
        +--> Citation validation
        |
        v
Cited answer / insufficient evidence
        |
        v
Streamlit UI
```

The workflow is intentionally compact enough for an individual portfolio project while demonstrating an end-to-end agentic RAG architecture.

Retrieval establishes the evidence set. The financial-analysis agent is instructed to use only retrieved evidence. When arithmetic is required, the model can call a restricted calculator through LangGraph. The final response is then checked against the retrieved citation markers before being displayed.

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

## Tech stack

- Python 3.11+
- Streamlit
- LangChain
- LangGraph
- Ollama
- Llama 3.2
- nomic-embed-text
- Chroma
- PyMuPDF
- Pydantic
- Pytest
- Ruff

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/srivarshapopuri1-arch/agentic-rag-financial-analyst.git
cd agentic-rag-financial-analyst
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -e ".[dev]"
```

### 4. Install Ollama

Install Ollama from its official distribution for your operating system.

Verify the installation:

```bash
ollama --version
```

### 5. Download the local models

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Verify them:

```bash
ollama list
```

### 6. Configure the application

Copy `.env.example` to `.env`.

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS/Linux:

```bash
cp .env.example .env
```

Default configuration:

```text
LLM_MODEL=llama3.2:3b
EMBEDDING_MODEL=nomic-embed-text
OLLAMA_BASE_URL=http://localhost:11434

CHROMA_DIR=chroma_db
COLLECTION_NAME=financial_documents
DATA_DIR=data

CHUNK_SIZE=1200
CHUNK_OVERLAP=200
TOP_K=6
LOG_LEVEL=INFO
```

No OpenAI API key is required.

## Financial documents

Place legitimate public financial-report PDFs in:

```text
data/
```

PDFs are deliberately ignored by Git.

Appropriate inputs include public annual reports, 10-Ks, 10-Qs, and other regulatory filings that you are permitted to use.

Do not add confidential employer/customer documents, credentials, or private financial information.

## Run

Make sure Ollama is running, then start the application:

```bash
streamlit run app.py
```

Open the Streamlit interface and select **Index documents**.

The application will:

1. Parse the PDFs.
2. Split them into chunks.
3. Generate local embeddings.
4. Store the chunks in Chroma.
5. Make the indexed evidence available to the LangGraph workflow.

Then ask financial questions through the Streamlit interface.

Example questions:

```text
What total revenue did the company report for the latest fiscal year? Cite the source.
```

```text
Compare revenue across the periods represented in these filings.
```

```text
What liquidity risks does management describe?
```

```text
Extract operating income and explain the reported year-over-year change.
```

Arithmetic questions can invoke the calculator tool when appropriate.

## Agentic workflow

### 1. Document research

The research step retrieves the most relevant chunks from Chroma using the user's question.

### 2. Financial analysis

The local Llama model analyzes the retrieved evidence and is instructed not to introduce unsupported outside facts.

### 3. Calculator tool

When arithmetic is required, the model can call a restricted calculator through the LangGraph workflow.

The calculator supports numeric arithmetic while rejecting names, function calls, imports, attributes, and other executable Python behavior.

### 4. Final answer generation

After analysis and any tool calls, the workflow constructs a final answer using the original question, retrieved evidence, and relevant tool results.

### 5. Grounding validation

A deterministic validation step checks that citations are present when required and that cited source/page markers belong to the retrieved evidence set.

If grounding fails, the application does not silently present the generated response as supported.

## Grounding behavior

The application is designed around three response cases:

**Supported question**

Returns an answer with source/page citations from retrieved evidence.

**Calculation question**

Can combine retrieved financial evidence with the calculator tool and return the calculation with documentary citations.

**Unsupported question**

Returns an explicit insufficient-evidence response rather than intentionally fabricating support.

Grounding validation improves reliability, but it does not prove that every interpretation or factual statement is semantically correct. Material conclusions should still be checked against the original filing.

## Configuration

`.env.example` contains the supported settings:

- `LLM_MODEL`
- `EMBEDDING_MODEL`
- `OLLAMA_BASE_URL`
- `CHROMA_DIR`
- `COLLECTION_NAME`
- `DATA_DIR`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `TOP_K`
- `LOG_LEVEL`

The default configuration uses:

```text
LLM_MODEL=llama3.2:3b
EMBEDDING_MODEL=nomic-embed-text
```

## Tests and code quality

Run the automated tests:

```bash
pytest
```

Run static checks:

```bash
ruff check .
```

The offline tests cover:

- document chunk metadata
- chunk configuration validation
- arithmetic calculations
- rejection of unsafe calculator expressions
- citation validation
- insufficient-evidence behavior

The live Ollama, embedding, Chroma, and Streamlit workflow requires the local Ollama service and downloaded models and is therefore separate from the offline unit tests.

## Verified local workflow

During local development, the application was successfully exercised through the following workflow:

```text
PDF
 -> PyMuPDF
 -> document chunks
 -> nomic-embed-text
 -> Chroma
 -> LangGraph
 -> Llama 3.2
 -> calculator tool when required
 -> grounded final response
 -> citation validation
 -> Streamlit
```

Local testing verified document indexing, retrieval-based financial answering, source/page citations, arithmetic analysis through the agent workflow, and insufficient-evidence handling.

## Limitations

- Image-only and scanned PDFs require OCR, which is not implemented.
- Complex financial tables can lose structure during plain PDF text extraction.
- Vector retrieval can miss relevant passages.
- Citation validation verifies source-marker usage; it is not a full semantic-entailment or hallucination detector.
- Cross-company and cross-period comparisons are only as complete as the indexed filings and retrieved evidence.
- Local inference speed depends on the user's CPU, GPU, RAM, and selected Ollama model.
- Smaller local models may be less reliable than larger models on complex financial reasoning.
- Ollama must be installed and running locally.
- This is a portfolio research assistant, not a production financial-data or investment-advice platform.

## Possible extensions

Potential future improvements include:

- SEC EDGAR API ingestion
- table-aware financial-document parsing
- retrieval reranking
- structured KPI extraction
- RAG evaluation datasets
- semantic grounding evaluation
- tracing and observability
- configurable local models
- FastAPI service layer

## Responsible use

Use public or appropriately licensed documents only.

Never commit:

- `.env`
- credentials or API keys
- private financial information
- confidential company documents
- local vector-database files

Always verify material financial conclusions against the original source filing.