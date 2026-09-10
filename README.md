# Local PDF RAG Pipeline

A small, local Retrieval-Augmented Generation (RAG) retrieval pipeline for
asking questions about one or more PDF books. It extracts readable PDF text,
creates overlapping chunks, embeds them with SentenceTransformers, stores the
vectors in a local FAISS index, and retrieves the most relevant passages for a
question.

The project supports retrieval evaluation and optional local LLM answer
evaluation. Ollama can generate grounded answers and judge them without an API
key or hosted model charges.

## Pipeline

```text
PDF
  -> cleaned text + detected headings
  -> structure-aware overlapping chunks
  -> embeddings + local FAISS vector database
  -> ranked relevant source chunks
```

## Features

- Local processing: no hosted vector database is required.
- PDF extraction that filters likely headers, footers, and small-font footnotes.
- Automatic heading candidates and section-aware chunking.
- Token-based chunks with paragraph-level overlap.
- SentenceTransformer embeddings using `all-MiniLM-L6-v2` by default.
- Exact FAISS search using normalized inner product (cosine-similarity ranking).
- Database validation before retrieval: index, metadata, and documents must agree.
- Optional JSON export of retrieved results.

## Requirements

- Python 3.10 or newer.
- A text-based PDF. Scanned/image-only PDFs need OCR before this pipeline can
  extract useful text.

Install dependencies in this project's virtual environment:

```powershell
python -m pip install -r requirements.txt
```

The first embedding run downloads the selected SentenceTransformer model.

## Configuration

Create a `.env` file in the repository root:

```dotenv
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
LOG_LEVEL=INFO
```

`EMBEDDING_DIMENSION` is a safety contract. It must match the selected model.
For `all-MiniLM-L6-v2`, the dimension is `384`. The build scripts stop if the
model produces a different dimension, preventing incompatible vector databases.

## Data storage structure

Keep source PDFs separate from generated artifacts:

```text
data/
├── source/                         # Original PDFs only
│   └── AI Engineering.pdf
├── cleaned/                        # Generated cleaned text and metadata
│   ├── AI Engineering.txt
│   └── AI Engineering.meta.json
├── chunks/                         # Generated chunk records
│   └── AI Engineering_chunks.jsonl
└── vector_db/                      # Generated searchable databases
    └── AI Engineering/
        ├── index.faiss             # FAISS vector index
        ├── documents.jsonl         # Chunk text and metadata in vector-ID order
        ├── metadata.json           # Model, dimension, metric, chunk count
        └── query_results.json      # Optional result export
```

`data/` is ignored by Git so books, generated vectors, and local experiment
outputs are not committed accidentally.

This separation supports reproducibility and data lineage:

```text
source PDF -> cleaned text -> chunks -> vector database
```

If chunks change, rebuild the vector database. Do not edit or reorder
`documents.jsonl` independently of `index.faiss`, because FAISS result IDs map
to documents by position.

## Quick start

1. Put a PDF in `data/source/`.

2. Build the database and immediately ask a question:

   ```powershell
   python src/run_pipeline.py "data\source\AI Engineering.pdf" --query "What is retrieval-augmented generation?" --top-k 3
   ```

3. The pipeline writes outputs to `data/cleaned/`, `data/chunks/`, and
   `data/vector_db/AI Engineering/`.

4. Ask additional questions without rebuilding the database:

   ```powershell
   python src/Retrieval.py "data\vector_db\AI Engineering" "How does chunk overlap help retrieval?" --top-k 5
   ```

5. Optionally save a retrieval result to JSON:

   ```powershell
   python src/Retrieval.py "data\vector_db\AI Engineering" "What is RAG?" --top-k 3 --save "data\vector_db\AI Engineering\rag_results.json"
   ```

## Running individual stages

The complete pipeline is recommended, but each stage can be run separately.

### 1. Extract cleaned text

```powershell
python src/convert_pdf_to_text.py "data\source\AI Engineering.pdf" -o "data\cleaned\AI Engineering.txt"
```

This also creates `data/cleaned/AI Engineering.meta.json`, which contains
PDF settings, page count, and detected heading candidates.

### 2. Create chunks

```powershell
python src/chunking.py "data\cleaned\AI Engineering.txt" -o "data\chunks\AI Engineering_chunks.jsonl" --chunk-size 400 --overlap 60
```

The chunker automatically looks for the matching `.meta.json` file unless
metadata lookup is disabled.

### 3. Build the vector database

```powershell
python src/embedding.py "data\chunks\AI Engineering_chunks.jsonl" -o "data\vector_db\AI Engineering"
```

### 4. Retrieve relevant chunks

```powershell
python src/Retrieval.py "data\vector_db\AI Engineering" "What is a vector database?" --top-k 3
```

## Main commands

```powershell
# Full pipeline: PDF -> text -> chunks -> vectors
python src/run_pipeline.py "data\source\book.pdf"

# Full pipeline with immediate retrieval
python src/run_pipeline.py "data\source\book.pdf" --query "Your question" --top-k 3

# Tune chunking
python src/run_pipeline.py "data\source\book.pdf" --chunk-size 400 --overlap 60

# Override output locations when needed
python src/run_pipeline.py "data\source\book.pdf" --cleaned-dir "other\cleaned" --chunks-dir "other\chunks" --output "other\vectors\book"
```

## Project modules

| File | Responsibility |
| --- | --- |
| `src/convert_pdf_to_text.py` | Extracts PDF body text, filters likely headers/footers/footnotes, and records heading/page metadata. |
| `src/chunking.py` | Splits cleaned text into overlapping JSONL chunk records. |
| `src/embedding_model.py` | Shared lazy-loading SentenceTransformer wrapper for document and query embeddings. |
| `src/vector_store.py` | FAISS vector-store persistence and JSONL validation. |
| `src/embedding.py` | Builds a local FAISS database from chunk JSONL. |
| `src/Retrieval.py` | Loads a saved database and retrieves ranked source chunks. |
| `src/run_pipeline.py` | Runs extraction, chunking, embedding, and optional retrieval in one command. |
| `src/config.py` | Loads shared environment configuration. |

## How retrieval works

The database contains three linked files:

```text
index.faiss       -> stored vectors
documents.jsonl   -> original chunks and metadata
metadata.json     -> model and index information
```

At query time:

1. The question is embedded with the same model stored in `metadata.json`.
2. FAISS searches for the `top_k` nearest chunk vectors.
3. Returned FAISS IDs are mapped to the corresponding JSONL chunk records.
4. The command prints the score, section, page, chunk ID, and source text.

The current index uses `faiss.IndexFlatIP`. With L2-normalized embeddings,
inner-product ranking is equivalent to cosine-similarity ranking. This exact
search strategy is a good fit for a book or a modest collection of books.

## Quality checks and troubleshooting

Before trusting answers, inspect the generated artifacts:

- Open samples from `data/cleaned/` and confirm headers, footers, and broken
  text are not polluting the book content.
- Inspect several JSONL chunks to confirm section boundaries and overlap are
  sensible.
- Test 15-30 questions with known answers and measure whether the correct
  passage appears in the top 3 or top 5 results.
- If a result is poor, investigate in this order: PDF extraction, chunking,
  source metadata, embedding model, then retrieval settings.

## Evaluate with a free local LLM

The default evaluator measures retrieval with RAGAS:

```powershell
python src/evaluate.py data/eval/golden_set.jsonl data/vector_db/AI_Engineering `
   -o data/eval/retrieval_results.json --top-k 3
```

For answer-level evaluation, install [Ollama](https://ollama.com/), start it,
and download a local model. A small model such as `llama3.2:3b` is a practical
starting point on a laptop:

```powershell
ollama pull llama3.2:3b
python src/evaluate.py data/eval/golden_set.jsonl data/vector_db/AI_Engineering `
   --llm-model llama3.2:3b -o data/eval/llm_results.json --top-k 3
```

For each golden-set question, the evaluator retrieves context, asks Ollama for
an answer with `[Context N]` citations, then asks the same local model to
return structured scores for:

- `correctness`: agreement with the reference answer
- `faithfulness`: whether material claims are supported by retrieved context
- `relevance`: whether the answer addresses the question

The output JSON contains aggregate scores and each generated answer. Use a
separate judge model with `--ollama-url` only if you run another local Ollama
server; the default URL is `http://localhost:11434`.

### Completed retrieval iteration

As a controlled experiment, `top_k` was changed from 3 to 5 while keeping the
golden set, FAISS database, embedding model, and judge fixed. Context recall
improved from `0.7305` to `0.8255`, while context precision changed from
`0.0445` to `0.0427`. The full before/after comparison is recorded in
[`data/eval/iteration_top_k_report.md`](data/eval/iteration_top_k_report.md).

Common issues:

| Issue | Likely cause | Action |
| --- | --- | --- |
| Very little extracted text | PDF is scanned/image-only | Run OCR before the pipeline. |
| `EMBEDDING_DIMENSION` mismatch | `.env` does not match the selected model | Update the model name or dimension, then rebuild the database. |
| Retrieval results are wrong | Index and metadata came from different builds | Rebuild the complete vector database directory. |
| Results lack source/book name | Chunk records do not yet include `source` metadata | Add source metadata before combining multiple books. |

## Current scope and next steps

This repository is intentionally a retrieval-focused RAG foundation. Good next
steps are:

1. Add `source`, `chapter`, and precise page-range metadata to chunks before
   combining multiple books.
2. Create a small evaluation set of real questions and measure Recall@3.
3. Add an answer-generation layer that cites retrieved passages.
4. Add metadata filters for chapter/book-specific questions.
5. Consider reranking only if evaluation shows that top-k retrieval often
   returns broadly related but incorrect passages.

For a single book or small library, keep `IndexFlatIP`. Consider HNSW or IVF
approximate indexes only after measured memory or latency problems at much
larger scale.
