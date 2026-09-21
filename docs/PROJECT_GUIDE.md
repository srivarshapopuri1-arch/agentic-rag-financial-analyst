# How the Project Works

I built this project to understand how retrieval and agent workflows can make long financial reports easier to work with. The main idea is to keep the answer tied to the report instead of relying on the model's general knowledge.

## Document processing

I start with public financial reports such as 10-Ks, 10-Qs, and annual reports. PyMuPDF reads the text one page at a time and keeps the filename and page number with the extracted text.

Financial reports can be long, so I split each page into smaller overlapping chunks. Keeping some overlap helps avoid losing context when an important sentence falls near a chunk boundary.

Each chunk keeps its source information:

```text
document name
page number
chunk id
text
```

## Embeddings and storage

I use `nomic-embed-text` through Ollama to create an embedding for each chunk. An embedding represents the meaning of the text as numbers, which makes it possible to search for passages that are related to a question even when the wording is different.

The chunks and embeddings are stored locally in Chroma.

When I index the same content again, the project creates deterministic IDs from the document information and text. This helps avoid adding duplicate copies of identical chunks.

## Asking a question

When I enter a question in the Streamlit interface, the project searches Chroma for the most relevant chunks. The default configuration retrieves six chunks.

Those retrieved chunks become the evidence for the rest of the workflow. The model is instructed to use that evidence instead of filling gaps from memory.

The flow looks like this:

```text
Financial reports
       |
       v
Extract text and page information
       |
       v
Split text into chunks
       |
       v
Create embeddings with Ollama
       |
       v
Store chunks in Chroma
       |
       v
Ask a question
       |
       v
Retrieve relevant chunks
       |
       v
Analyze retrieved evidence
       |
       +------ calculation needed ------> calculator
       |                                     |
       |<------------------------------------+
       v
Create final answer
       |
       v
Check citations
       |
       v
Answer with sources or insufficient evidence
```

## LangGraph workflow

LangGraph connects the different parts of the question-answering process.

### Document research

The first step searches Chroma using the question and collects the relevant passages. If nothing useful is available, the workflow can stop with an insufficient-evidence response.

### Financial analysis

The retrieved passages are sent to the local Llama model. The prompt tells the model to use only the supplied evidence and to avoid inventing figures, periods, companies, or conclusions.

### Calculator

Some financial questions need arithmetic, such as percentage changes between two reported values. For those questions, the model can call the calculator.

I kept the calculator restricted to basic numeric arithmetic. It does not allow arbitrary Python execution, imports, function calls, names, or attributes.

The calculator only handles the calculation. The numbers used in the calculation still need to come from the financial report.

### Final answer

After the analysis and any calculation, the model creates the final response using the original question, retrieved passages, and calculator result.

A citation looks like:

```text
[report.pdf p. 42]
```

This makes it possible to go back to the original page and check the information.

### Citation validation

The last step checks whether the answer contains citations and whether those source/page markers were actually part of the retrieved evidence.

If the answer refers to evidence that was not retrieved, the validation fails. If the available documents do not support the answer, the workflow returns an insufficient-evidence response.

This check helps prevent unsupported source references, but it does not prove that every sentence is completely supported by the cited passage.

## Streamlit interface

The Streamlit app gives me two main actions.

**Index documents** reads the PDFs from the `data/` directory, splits them into chunks, creates embeddings, and stores them in Chroma.

**Analyze** takes a question, runs the LangGraph workflow, and displays the answer. I can also open the retrieved-evidence section to see the passages that were supplied to the model.

## Main files

`app.py` contains the Streamlit interface.

`src/financial_analyst/ingestion.py` reads PDFs and splits their text into chunks while preserving source and page information.

`src/financial_analyst/vectorstore.py` creates Ollama embeddings, stores document chunks in Chroma, and retrieves relevant passages.

`src/financial_analyst/graph.py` defines the LangGraph flow for retrieval, analysis, calculator calls, final answer generation, and validation.

`src/financial_analyst/agents.py` creates the local Llama model, formats retrieved evidence, and validates citation markers.

`src/financial_analyst/tools.py` contains the restricted calculator.

`src/financial_analyst/config.py` loads the project settings.

`src/financial_analyst/logging_config.py` configures logging.

`tests/` contains tests for document chunking, calculator restrictions, citations, and insufficient-evidence behavior.

## Running it

After installing the dependencies and the Ollama models described in the README, I place public financial PDFs in:

```text
data/
```

Then I start the application with:

```bash
streamlit run app.py
```

I index the documents from the sidebar before asking questions.

The default local models are:

```text
llama3.2:3b
nomic-embed-text
```

`llama3.2:3b` handles the analysis and response generation. `nomic-embed-text` creates the embeddings used for document retrieval.

## Testing

The automated checks are:

```bash
pytest
ruff check .
```

The tests cover the parts of the project that can be checked without running a local model, including chunk metadata, chunk settings, arithmetic restrictions, citation validation, and insufficient-evidence handling.

I also check the complete flow locally with Ollama and Streamlit by indexing a public report and trying questions that require direct retrieval, a calculation, and information that is not available in the document.

## Data and local files

The financial PDFs and Chroma database stay local. Environment files, virtual environments, caches, logs, generated package files, and local document data are excluded from the repository.

I use public or appropriately licensed reports and avoid putting credentials, confidential documents, or private financial information in the project.

## Limitations

The current PDF extraction works best when the report contains extractable text. Scanned reports would need OCR, which I have not added.

Complex financial tables can lose their original structure when they are extracted as plain text. Retrieval is based on similarity, so it can also miss a relevant passage.

The citation validator checks that a citation came from the retrieved evidence, but it does not perform a full semantic verification of every claim.

For now, I have kept those limitations visible instead of adding more layers to the project. The current structure gives me a clear way to experiment with retrieval, tool use, citations, and local language models while still being able to follow the complete flow in the code.
