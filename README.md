# Agentic RAG Financial Analyst

I built this project to explore a question I kept coming back to: how useful can RAG and simple AI-agent workflows be when reading long financial reports?

The project is a small local-first research assistant for public financial documents such as 10-Ks, 10-Qs, and annual reports. It parses PDFs, retrieves relevant passages, lets an LLM reason over that evidence, can call a restricted calculator, and checks that final answers cite retrieved source pages. The goal is not to automate financial judgment, but to make document exploration easier while keeping the evidence visible.

> This is an experimental document-analysis project, not investment advice.

## What it does

- Reads public financial-report PDFs with PyMuPDF.
- Preserves source and page metadata during chunking.
- Creates local embeddings with `nomic-embed-text` through Ollama.
- Stores and retrieves chunks with Chroma.
- Uses `llama3.2:3b` locally for document-grounded analysis.
- Coordinates retrieval, analysis, calculator tool calls, final answer generation, and validation with LangGraph.
- Includes a restricted arithmetic calculator for financial calculations.
- Requires source/page citations for grounded answers.
- Rejects citations that are outside the retrieved evidence set.
- Returns an explicit insufficient-evidence response when it cannot support an answer.
- Provides a simple Streamlit interface.
- Includes configuration, logging, error handling, tests, and secret-safe defaults.

## How it works

```text
Financial PDFs
      |
      v
PyMuPDF parsing
      |
      v
Page-aware chunking
      |
      v
nomic-embed-text (Ollama)
      |
      v
Chroma vector store
      |
      v
User question
      |
      v
LangGraph
  |-- document research / retrieval
  |-- financial analysis
  |-- calculator tool when needed
  |-- final answer generation
  `-- citation validation
      |
      v
Cited answer or insufficient evidence
      |
      v
Streamlit
```

The workflow stays deliberately small. Retrieval establishes the evidence available to the model. The analysis step is instructed to use only that evidence. If arithmetic is needed, the model can call the calculator tool and then continue reasoning with the result. A final generation step produces the response, and deterministic validation checks whether its citation markers belong to the retrieved evidence.

The validation step does **not** prove that every sentence is factually entailed by its citation. It is a guardrail around source use, not a complete hallucination detector.

For a plain-language walkthrough of the RAG flow and the Git workflow used for this repository, see [`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md).

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

Python 3.11+, LangChain, LangGraph, Ollama, Llama 3.2, nomic-embed-text, Chroma, PyMuPDF, Streamlit, Pydantic, Pytest, and Ruff.

## Setup

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/srivarshapopuri1-arch/agentic-rag-financial-analyst.git
cd agentic-rag-financial-analyst
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

Install the project and development dependencies:

```bash
pip install -e ".[dev]"
```

Install Ollama, then download the two local models used by the default configuration:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
ollama list
```

Copy the example configuration:

```powershell
Copy-Item .env.example .env
```

On macOS/Linux:

```bash
cp .env.example .env
```

The default settings are:

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

## Add financial documents

Place public financial-report PDFs in:

```text
data/
```

PDFs in this directory are ignored by Git. Do not add confidential documents, credentials, private financial information, or other material that should not be public.

## Run

Make sure Ollama is running, then start Streamlit:

```bash
streamlit run app.py
```

In the sidebar, select **Index documents**. The application parses the PDFs, chunks the text, creates embeddings, and stores the chunks in Chroma. After indexing, ask questions in the main interface.

## Example questions

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

```text
Calculate the percentage change between two reported revenue figures and cite the source values.
```

The quality of an answer depends on whether the relevant passages are present in the indexed documents and retrieved for the question.

## Agents and tools

### Document research

The first LangGraph node retrieves the most relevant Chroma chunks for the question. Those chunks form the evidence set used by the rest of the workflow.

### Financial analysis

The analysis node uses the local Llama model and is instructed to reason only from retrieved evidence. It can request the calculator when arithmetic is required.

### Calculator

The calculator evaluates a deliberately small subset of numeric arithmetic. It rejects names, imports, function calls, attributes, and other executable Python behavior.

### Final answer

After analysis and any calculator calls, the graph asks the model for a concise final response using the original question, retrieved evidence, and tool results.

### Validation

A deterministic validation step checks citation presence and verifies that cited source/page markers occur in the retrieved evidence. If that check fails, the application does not present the generated text as a grounded answer.

## Tests

Run:

```bash
pytest
ruff check .
```

The current tests cover chunk metadata, chunk configuration validation, arithmetic, rejection of unsafe calculator expressions, citation validation, and insufficient-evidence behavior.

The live Ollama/Chroma/Streamlit path is separate from the offline unit tests because it requires local models and a running Ollama service.

## Experiments

While building the project, I used a public sample financial filing locally to exercise three behaviors rather than treating generated text alone as a successful test:

1. A question whose answer was present in the filing, to check retrieval and page citation behavior.
2. A percentage-change question, to check that arithmetic could go through the calculator while the source figures remained tied to document evidence.
3. A question asking for information not supported by the indexed filing, to check the insufficient-evidence path.

These checks were used to debug the workflow itself. The sample PDF and its financial values are intentionally not committed, so this repository does not claim those local figures as reproducible results.

## Observations

A few things became clear during the experiment:

- Page metadata is useful because it makes retrieved evidence easier to inspect than a citation to a document name alone.
- Tool use and source grounding are different concerns. A calculator can verify arithmetic, but the numeric inputs still need documentary support.
- Deterministic citation checks are useful as a final guardrail, but they cannot determine whether a sentence truly follows from the cited passage.
- Smaller local models make the project inexpensive to run, but complex financial reasoning can be less consistent and local inference speed depends heavily on hardware.
- Retrieval quality matters as much as generation quality. If the relevant passage is not retrieved, the model should not fill the gap from memory.

These are implementation observations from working on this project, not financial conclusions or benchmark claims.

## Limitations

- Scanned or image-only PDFs require OCR, which is not implemented.
- Complex tables can lose structure during plain PDF text extraction.
- Similarity search can miss relevant passages.
- The current workflow does not perform table-aware extraction or normalized financial-statement modeling.
- Citation validation checks citation markers, not semantic entailment.
- Cross-company and cross-period comparisons are only as complete as the indexed documents and retrieved evidence.
- Smaller local models may be inconsistent on difficult reasoning tasks.
- Ollama must be installed and running locally.
- The project is for document exploration and experimentation, not financial advice.

## Ideas to explore next

Possible extensions include SEC EDGAR ingestion, table-aware parsing, retrieval reranking, structured metric extraction, evaluation datasets, semantic grounding checks, tracing, and a small API layer.

## Data and secrets

Use public or appropriately licensed documents only.

The repository ignores local PDFs, the Chroma database, `.env`, virtual environments, logs, and generated package metadata. Never commit API keys, credentials, confidential documents, or private financial information.
