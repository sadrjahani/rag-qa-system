# Local PDF RAG with Ollama and FAISS

A transparent, local Retrieval-Augmented Generation (RAG) pipeline for asking grounded questions about text-based PDF documents. It extracts PDF text, creates structure-aware chunks, embeds them with SentenceTransformers, stores them in a local FAISS index, retrieves relevant passages, and can generate an answer with a local Ollama model.

No hosted vector database or cloud LLM is required.

## What it does

```text
PDF
  -> cleaned text and heading metadata
  -> overlapping, section-aware chunks
  -> SentenceTransformer embeddings
  -> local FAISS vector database
  -> top-k retrieved passages
  -> optional grounded answer from Ollama
```

The project is deliberately modular: PDF extraction, chunking, embedding, storage, retrieval, generation, and evaluation can be run and inspected independently.

## Features

- Local PDF extraction and noise filtering with PyMuPDF.
- Section-aware, overlapping chunking with JSONL output.
- Local `all-MiniLM-L6-v2` embeddings by default.
- Exact FAISS search using cosine-similarity ranking.
- Persisted FAISS index, aligned chunk records, and build metadata.
- Integrity checks for embedding dimension and index/document alignment.
- Local grounded answer generation through Ollama.
- Retrieval and generation evaluation utilities.

## Requirements

- Python 3.10 or newer.
- A text-based PDF. Run OCR first for scanned or image-only PDFs.
- [Ollama](https://ollama.com/) only if you want answer generation or generation evaluation. Retrieval and indexing do not require Ollama.

Install the Python dependencies in a virtual environment:

```powershell
python -m pip install -r requirements.txt
```

For local answer generation, download the default model:

```powershell
ollama pull phi4-mini
```

The first embedding run also downloads the selected SentenceTransformer model.

## Configuration

Copy the example file and adjust it only when needed:

```powershell
Copy-Item .env.example .env
```

```dotenv
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
LOG_LEVEL=INFO
GENERATION_MODEL=phi4-mini
```

`EMBEDDING_DIMENSION` is validated against the actual output dimension of the embedding model. If you change the embedding model, update this value to match or remove it and rebuild the vector database. An index must always be queried with the same embedding model that created it.

## Data layout

Keep inputs and generated artifacts under `data/`:

```text
data/
├── source/       # Original PDFs
├── cleaned/      # Extracted text and PDF metadata
├── chunks/       # Chunk JSONL files
└── vector_db/    # One FAISS database directory per PDF
```

For a source file named `data/source/book.pdf`, the default pipeline outputs are:

```text
data/cleaned/book.txt
data/cleaned/book.meta.json
data/chunks/book_chunks.jsonl
data/vector_db/book/
├── index.faiss
├── documents.jsonl
└── metadata.json
```

`data/`, `results/`, `.env`, and generated indexes are ignored by Git. Do not commit copyrighted books, embeddings, or private evaluation data. Also do not manually reorder `documents.jsonl`: its line order is aligned with FAISS vector IDs.

## Quick start

1. Put a PDF in `data/source/`.

2. Build a vector database and retrieve passages in one command:

```powershell
python src/run_pipeline.py "data\source\book.pdf" --query "What is retrieval-augmented generation?" --top-k 3
```

3. Query the saved database without rebuilding it:

```powershell
python src/retrieval.py "data\vector_db\book" "How does chunk overlap help retrieval?" --top-k 3
```

Example retrieval output:

```text
Results for: How does chunk overlap help retrieval?

1. Score: 0.7214
   Section: Chunking strategy
   Page: 12
   Chunk ID: 8
   Text: ...
```

4. Generate an answer grounded in retrieved passages (requires Ollama):

```powershell
python src/generation.py "data\vector_db\book" "What is retrieval-augmented generation?" --top-k 3
```

The generator instructs the local model to use only supplied context and cite context numbers. Always inspect the returned sources before relying on an answer.

## Running individual stages

```powershell
# Extract text and PDF metadata
python src/convert_pdf_to_text.py "data\source\book.pdf" -o "data\cleaned\book.txt"

# Create chunks
python src/chunking.py "data\cleaned\book.txt" -o "data\chunks\book_chunks.jsonl" --chunk-size 400 --overlap 60

# Build a FAISS database
python src/embedding.py "data\chunks\book_chunks.jsonl" -o "data\vector_db\book"

# Retrieve relevant chunks
python src/retrieval.py "data\vector_db\book" "What is RAG?" --top-k 3
```

Use `run_pipeline.py` for the normal workflow. Individual commands are useful when debugging a particular stage or experimenting with chunking settings.

## How retrieval works

The database contains three linked artifacts:

```text
index.faiss       -> normalized embedding vectors
documents.jsonl   -> chunk text and metadata in vector-ID order
metadata.json     -> model name, dimension, metric, and document count
```

The question is embedded with the model recorded in `metadata.json`. FAISS performs exact inner-product search over L2-normalized vectors, which is equivalent to cosine-similarity ranking. Returned vector IDs are mapped back to the corresponding JSONL chunk records.

`IndexFlatIP` is an appropriate, simple choice for a book or small collection of books. Consider approximate indexes only after measuring memory or latency problems on a much larger corpus.

## Evaluation

Evaluation data is project-specific and intentionally not included in this repository. A golden-set JSONL record needs at least a `question` and one or more `reference_contexts`.

Run retrieval evaluation without an LLM judge:

```powershell
python src/evaluate.py "data\eval\golden_set.jsonl" "data\vector_db\book" --mode retrieval --top-k 3 -o "results\retrieval_results.json"
```

Run local generation evaluation with Ollama:

```powershell
python src/evaluate.py "data\eval\golden_set.jsonl" "data\vector_db\book" --mode generation --model phi4-mini --top-k 3 -o "results\generation_results.json"
```

In one controlled retrieval experiment on a 68-question internal golden set, increasing `top_k` from 3 to 5 increased context recall from `0.7305` to `0.8255`; context precision changed from `0.0445` to `0.0427`. These values describe that specific corpus and dataset, not a universal project score.

## Project modules

| Module | Responsibility |
| --- | --- |
| `src/convert_pdf_to_text.py` | PDF extraction, cleaning, heading and page metadata |
| `src/chunking.py` | Structure-aware chunking and JSONL output |
| `src/embedding_model.py` | Lazy-loaded SentenceTransformer embedding backend |
| `src/vector_store.py` | FAISS persistence and consistency checks |
| `src/embedding.py` | Vector database construction from chunk JSONL |
| `src/retrieval.py` | FAISS loading, query embedding, and ranked retrieval |
| `src/generation.py` | Grounded local answer generation through Ollama |
| `src/evaluate.py` | Retrieval and generation evaluation entry point |
| `src/run_pipeline.py` | End-to-end PDF-to-vector-database pipeline |

## Limitations and next steps

- Chunk sizes are measured with `tiktoken`, while the embedding model has its own tokenizer. Before indexing a large corpus, calibrate chunk size against the embedding model's maximum input length to avoid truncation.
- Chunk records should include reliable `source`, `chapter`, and page-range metadata before combining multiple books.
- Evaluation metrics are meaningful only when the dataset, corpus, and command are documented together.
- Add automated tests and continuous integration before treating the project as a production service.

## License

This project is licensed under the [MIT License](LICENSE).
