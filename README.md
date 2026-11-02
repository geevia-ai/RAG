# Kubernetes Documentation RAG

## Overview

I started this project to build a Retrieval-Augmented Generation (RAG) system using Kubernetes documentation as the source of knowledge.

The goal is to take Markdown documentation, turn it into searchable vector representations, retrieve the most relevant sections for a question, and use an LLM to generate an answer grounded in those sections.

The project started with a basic document ingestion and retrieval pipeline. I am improving it incrementally as I encounter limitations and make implementation decisions.

## Current pipeline

```text
Kubernetes Markdown documentation
              |
              v
       Content cleanup
              |
              v
        Token chunking
              |
              v
       Generate embeddings
              |
              v
       Store in JSONL
              |
              v
        User question
              |
              v
      Generate query embedding
              |
              v
       Similarity search
              |
              v
       Retrieve top chunks
              |
              v
        LLM generates answer
```

## Project structure

```text
.
├── docs/
│   └── Kubernetes Markdown files
├── vectors/
│   └── vectors.jsonl
├── RAG_save_document_vectors.py
└── RAG_query.py
```

## Implementation stages

| Stage | What it does |
|---|---|
| 1. Collect documents | Reads Kubernetes Markdown files from `docs/`. |
| 2. Clean content | Removes documentation markup that is not useful as context while preserving meaningful text. |
| 3. Chunk documents | Splits documents into token-based chunks with overlap. |
| 4. Generate embeddings | Uses `all-MiniLM-L6-v2` to convert each chunk into a vector. |
| 5. Store vectors | Saves the chunk text, source metadata, and embedding in a JSONL file. |
| 6. Retrieve context | Embeds the user's question and retrieves the top matching chunks using similarity scores. |
| 7. Generate an answer | Sends the question and retrieved text to the LLM to generate a grounded response. |

## Engineering decisions

This table summarizes the main problems I encountered and the changes I made. Each decision links to the detailed reasoning below.

| Decision | Problem | Initial approach | New approach | Why | Trade-offs | Validation | Details |
|---|---|---|---|---|---|---|---|
| 001 | Word counts do not reflect model token limits. | Split documents by word count. | Split using the model tokenizer. | Keep chunks within the model's input limit. | More control, but tokenizer-specific. | Re-tokenize chunks and reject oversized ones. | [Details](#decision-001-token-based-chunking) |
| 002 | Chunk boundaries can separate related context. | Create independent chunks. | Add a 50-token overlap. | Preserve context between neighboring chunks. | More chunks and repeated text. | Inspect neighboring chunks for shared context. | [Details](#decision-002-overlapping-chunks) |
| 003 | Raw Kubernetes Markdown contains markup and formatting noise. | Use the source text as-is. | Clean comments, shortcodes, and links while preserving useful text. | Give the embedding model cleaner content. | Cleanup rules are specific to the source format. | Inspect cleaned output against the original. | [Details](#decision-003-cleaning-the-source-content) |
| 004 | Similarity scores need a consistent vector scale. | Generate embeddings without normalization. | Normalize document and query embeddings. | Use dot product as cosine similarity. | Vector magnitude is no longer retained. | Confirm both sides use normalized embeddings. | [Details](#decision-004-normalized-embeddings) |
| 005 | Generated vectors need to persist between runs. | Keep generated data in memory. | Store each chunk as a JSONL record. | Reuse embeddings without regenerating them every time. | Simple to inspect, but not optimized for large-scale search. | Reload records and verify their contents. | [Details](#decision-005-persisting-vectors-in-jsonl) |
| 006 | Comparing each vector individually would add unnecessary Python-loop overhead. | Iterate through vectors to calculate scores. | Use NumPy matrix multiplication. | Calculate similarity scores in one vectorized operation. | All vectors are loaded into memory. | Inspect the highest-scoring retrieved chunks. | [Details](#decision-006-vectorized-similarity-search) |
| 007 | Decoding token IDs back into text did not preserve the original source text exactly. | Decode each token window using `tokenizer.decode()`. | Use `return_offsets_mapping=True` to extract chunks directly from the original text. | Preserve original casing and formatting without decoding and recreating the text. | Token boundaries can split words, causing the extracted text to produce a different token count when re-tokenized. | Re-tokenize extracted chunks and verify they stay within 256 tokens. Testing identified a chunk that became 257 tokens. | [Details](#decision-007-offset-based-text-extraction) |
---

## Decision details

### Decision 001: Token-based chunking

**Problem**

The embedding model has a maximum input length of 256 tokens. Splitting text by word count does not guarantee that a chunk stays within this limit because a single word can be represented by multiple tokens.

**Initial approach**

I initially split the document using a fixed number of words per chunk.

**Next approach**

I use the model's tokenizer to split the document into token windows. The current implementation sets the window size to `model.max_seq_length - 5` and checks the decoded chunks again before generating embeddings.

**Why**

The tokenizer determines how the model actually sees the text. Using it for chunking makes the chunk size more reliable than counting words.

**Trade-offs**

The implementation is now tied to the tokenizer and model being used. It also requires converting token windows back into text before generating embeddings.

**Validation**

The script tokenizes each generated chunk again and raises an error if its token count exceeds the model's maximum sequence length.

**New approach**

I use the model's tokenizer to split documents into token windows, using `model.max_seq_length` as the maximum input size. I then use offset mapping to extract the original text for each chunk instead of decoding token IDs back into text. See Decision 007 for more details.

### Decision 002: Overlapping chunks

**Problem**

When documents are split into independent chunks, related information can end up on opposite sides of a chunk boundary.

**Initial approach**

Chunks were created without shared content.

**New approach**

I added a 50-token overlap using the tokenizer's `stride` parameter.

**Why**

The overlap gives neighboring chunks some shared context. This can help retrieval when an answer depends on information near a chunk boundary.

**Trade-offs**

Overlap increases the number of chunks, stored embeddings, and repeated content.

**Validation**

Inspect neighboring chunks and confirm that they share the expected context without losing document content.

### Decision 003: Cleaning the source content

**Problem**

Kubernetes documentation contains Hugo shortcodes, HTML comments, Markdown links, and other formatting that is not always useful when generating embeddings.

**Initial approach**

Use the Markdown source directly.

**New approach**

I added a cleanup function that:
- Preserves glossary tooltip text.
- Removes HTML comments.
- Removes Hugo shortcode wrapper tags.
- Preserves Markdown link text while removing the URL.

**Why**

The embedding model should receive meaningful documentation text rather than unnecessary markup.

**Trade-offs**

The cleanup rules are designed around the current documentation source. Other Markdown repositories may require different cleanup rules.

**Validation**

Inspect the cleaned text to confirm that useful content is preserved and unwanted markup is removed.

### Decision 004: Normalized embeddings

**Problem**

The retrieval process needs a consistent way to compare document embeddings with a query embedding.

**Initial approach**

Generate embeddings without explicitly normalizing them.

**New approach**

I use `normalize_embeddings=True` when generating both document and query embeddings.

**Why**

When vectors are normalized, their dot product is equivalent to cosine similarity. This allows similarity scores to be calculated using a simple matrix multiplication.

**Trade-offs**

Normalization removes vector magnitude. This is appropriate for the cosine-similarity approach used in this project.

**Validation**

Confirm that document and query embeddings are both generated with normalization enabled.

### Decision 005: Persisting vectors in JSONL

**Problem**

Keeping generated embeddings only in memory means they need to be generated again whenever the ingestion script runs.

**Initial approach**

Store generated data in Python objects during execution.

**New approach**

Save each chunk as a separate JSONL record containing:

```json
{
  "doc_id": "concepts/workloads",
  "chunk_id": "concepts/workloads#chunk-1",
  "source": "docs/concepts/workloads.md",
  "text": "The original chunk text...",
  "embedding": [0.012, -0.034, 0.056]
}
```

**Why**

JSONL provides a simple persistent format. Each record keeps the chunk text and its embedding together with the source metadata needed for retrieval and citations.

**Trade-offs**

The format is easy to inspect and work with locally, but it does not provide an index for efficient large-scale vector search. The current ingestion script also rebuilds the vector file rather than updating individual records.

**Validation**

Run ingestion, inspect the generated JSONL records, and load them again from the query script.

### Decision 006: Vectorized similarity search

**Problem**

The query needs to be compared against all stored document embeddings to find the most relevant chunks.

**Initial approach**

Calculate similarity scores by iterating over the stored vectors.

**New approach**

I load the embeddings into a NumPy array and calculate the scores using matrix multiplication:

```python
scores = vectors @ query_vector
```

Then I sort the scores in descending order and select the top three matches.

**Why**

NumPy performs the calculation as a vectorized operation rather than requiring a Python loop for every embedding.

**Trade-offs**

This approach is simple and effective for the current dataset, but all embeddings must fit in memory. It also performs a brute-force comparison against every stored vector.

**Validation**

Inspect the highest-scoring chunks for a set of test questions and confirm that the retrieved text is relevant to the question.

### Decision 007: Offset-based text extraction

**Problem**

The initial implementation decoded token IDs back into text to create chunks. However, decoding did not preserve the original source text exactly. For example, the decoded text could change the original casing and formatting.

**Initial approach**

Use `tokenizer.decode()` on each token window to convert the token IDs back into text.

**New approach**

Use `return_offsets_mapping=True` when tokenizing the original document.

The offset mapping provides the character start and end positions for each token. I use these positions to extract the chunk directly from the original document using Python string slicing.

**Why**

This preserves the original text, including its casing and formatting, without relying on a decode-and-re-tokenize process.

**Trade-offs**

Token boundaries do not necessarily align with complete word boundaries. A chunk can start or end in the middle of a word.

When the extracted text is tokenized again, the tokenizer may generate a different number of tokens than the original window.

**Validation**

I re-tokenize every extracted chunk with `add_special_tokens=True` and `truncation=False`, then verify that it does not exceed the model's 256-token limit.

During testing, I found a chunk that started with `eration`. In the original token window, this was represented by one token (`16754`). After extracting and re-tokenizing the substring, it became two tokens (`3690` and `3508`), resulting in 257 tokens instead of 256.

The validation correctly identifies this condition and raises an error rather than allowing an oversized chunk to be embedded.

Whole-word boundary handling is a potential improvement that I will evaluate separately.

---

## Setup and execution

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install sentence-transformers numpy openai
```

### 3. Add the source documents

Place the Kubernetes Markdown files under the `docs/` directory.

### 4. Generate document embeddings

```bash
python RAG_save_document_vectors.py
```

This reads the Markdown files, cleans the content, creates chunks and embeddings, and writes the records to `vectors/vectors.jsonl`.

### 5. Configure the OpenAI API key

The query script reads the key from the `API_KEY` environment variable.

```bash
export API_KEY="your-api-key"
```

### 6. Ask a question

```bash
python RAG_query.py
```

Enter a question when prompted. The script generates a query embedding, retrieves the top matching chunks, and sends the retrieved context to the LLM.

## Next improvements

I am keeping future improvements separate from implemented engineering decisions. As I implement and validate each one, I will add it to the decision history above.

