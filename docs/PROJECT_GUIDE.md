# Project Guide

This note explains the project in plain language and records the Git workflow used to keep the repository tidy. The README is the main setup reference; this guide focuses on understanding what happens after a PDF is added and why the repository is organized the way it is.

## The project in simple terms

The application lets me put public financial-report PDFs into a local folder and ask questions about them.

Instead of asking the language model to answer from general knowledge, the application first searches the reports for passages related to the question. Those passages become the evidence for the answer. The model is told to use only that evidence and to cite the document and page it used.

For questions that require arithmetic, the model can use a small calculator tool. The calculator handles the math, while the financial values still need to come from the retrieved document evidence.

At the end, the application checks the citation markers. If the answer cannot be supported by the retrieved evidence, it should return an insufficient-evidence response rather than fill in missing information.

## What happens when documents are indexed

1. **Read the PDFs.** PyMuPDF extracts text from each non-empty page and keeps the PDF filename and page number.
2. **Split the text.** Long pages are divided into overlapping chunks so retrieval can work with smaller pieces of text. The source and page metadata stay attached.
3. **Create embeddings.** Ollama runs the local `nomic-embed-text` model to turn each chunk into a numeric representation of its meaning.
4. **Store the chunks.** Chroma saves the embeddings and text locally. Deterministic chunk IDs help avoid creating duplicate copies when the same content is indexed again.

The PDFs and local Chroma database are intentionally ignored by Git.

## What happens when a question is asked

The LangGraph workflow is:

```text
question
   |
   v
retrieve relevant chunks from Chroma
   |
   v
analyze only the retrieved evidence
   |
   +---- arithmetic needed? ---- yes ---> calculator
   |                                  |
   |                                  +----> analysis continues
   v
generate final answer
   |
   v
validate citation markers
   |
   v
cited answer OR insufficient evidence
```

### Retrieval

The question is embedded and Chroma returns the most similar chunks. By default, the application asks for the top six chunks.

Retrieval is important because these chunks define what the model is allowed to use. If an important passage is not retrieved, the model is instructed not to invent the missing information.

### Analysis and tool use

The local Llama model receives the question and retrieved evidence. LangGraph coordinates this step.

If the model decides arithmetic is needed, it can call the calculator. The calculator accepts only basic numeric arithmetic. It does not run arbitrary Python code, imports, names, or function calls.

After a calculator call, the result goes back into the analysis flow before the final response is written.

### Final answer and validation

The final answer is generated from the original question, retrieved evidence, and any calculator result. Financial claims are expected to use citation markers such as:

```text
[report.pdf p. 42]
```

The validator checks that citations are present and that the cited source/page markers belong to the evidence retrieved for that question. This is a useful guardrail, but it does not prove that every sentence is semantically supported by the cited passage.

## Main files

- `app.py` — Streamlit interface for indexing documents and asking questions.
- `src/financial_analyst/ingestion.py` — PDF reading and text chunking.
- `src/financial_analyst/vectorstore.py` — Ollama embeddings, Chroma storage, indexing, and retrieval.
- `src/financial_analyst/graph.py` — LangGraph workflow connecting retrieval, analysis, calculator use, finalization, and validation.
- `src/financial_analyst/agents.py` — local LLM setup, evidence formatting, and citation validation.
- `src/financial_analyst/tools.py` — restricted calculator.
- `src/financial_analyst/config.py` — settings loaded from environment variables.
- `src/financial_analyst/logging_config.py` — logging setup.
- `tests/` — offline tests for ingestion, calculator behavior, and citation validation.
- `.env.example` — example local configuration without secrets.
- `.gitignore` — keeps local data, model indexes, environments, caches, logs, and secrets out of Git.

## Running the project

After following the installation steps in the README, put public financial PDFs in `data/`, make sure Ollama is running, and start:

```bash
streamlit run app.py
```

Use **Index documents** before asking questions. Indexing builds or updates the local Chroma collection.

The two default Ollama models are:

```text
llama3.2:3b
nomic-embed-text
```

The first generates and reasons over answers. The second creates embeddings for retrieval.

## Checking changes

The normal code checks are:

```bash
pytest
ruff check .
```

The tests are intentionally offline. They do not require a running Ollama server or a committed financial PDF.

A live check is separate: run Ollama and Streamlit, index a public report, then try a supported question, a calculation question, and a question whose answer is not in the document.

## Git workflow used for this repository

The repository now uses `main` as the single long-lived branch. Short-lived branches can still be useful when making a change, but completed work should be reviewed and merged back into `main`.

A simple workflow for a future change is:

```bash
git checkout main
git pull origin main
git checkout -b short-description
```

Make the change, run the checks, then inspect what changed:

```bash
git status
git diff
pytest
ruff check .
```

Commit and push the branch:

```bash
git add .
git commit -m "Describe the change"
git push -u origin short-description
```

After the branch is reviewed and merged on GitHub, update the local copy:

```bash
git checkout main
git pull origin main
```

When the merged branch no longer contains unique work, remove it locally and remotely:

```bash
git branch -d short-description
git push origin --delete short-description
git fetch --prune
```

Before deleting a branch, compare or review it first. The important rule is that unique work should be merged or intentionally preserved before the branch is removed.

## What Git should not contain

Local runtime material does not belong in the repository. The current `.gitignore` excludes items such as:

- `.env` files other than the safe example
- local PDFs in `data/`
- the Chroma database
- virtual environments
- Python, pytest, Ruff, mypy, and notebook caches
- logs and coverage output
- editor/OS metadata
- generated package metadata

API keys, credentials, confidential reports, and private financial information should never be committed.

## Current boundaries

This project works with text that PyMuPDF can extract from PDFs. It does not implement OCR for scanned reports or table-aware financial-statement parsing. Retrieval is similarity based, so relevant passages can occasionally be missed. The citation check verifies that cited markers came from retrieved evidence; it is not a complete factual-entailment system.

Those limits are intentional for now. The project stays small enough that the RAG flow, agent/tool behavior, and grounding checks can be understood directly from the code.
