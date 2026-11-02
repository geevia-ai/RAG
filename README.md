# Building a RAG System from Scratch

This repository documents my journey of building a Retrieval-Augmented Generation (RAG) system using Python, local embedding models, and OpenAI.

The objective is to understand how the different components of a RAG system work, starting with a basic implementation and progressively improving it.

Rather than relying on frameworks that hide the underlying implementation, I'm building the individual components myself. Along the way, I'm documenting the problems I encounter, the approaches I try, the decisions I make, and the improvements.

The knowledge source for this project is the public Kubernetes documentation.

## 1. Project overview

The application allows users to ask questions about a collection of documents.

For example:

**Question:** What manages ReplicaSets in Kubernetes?

The application:

1. Searches the Kubernetes documentation for relevant content.
2. Identifies the most relevant document chunks.
3. Sends those chunks, together with the original question, to a language model.
4. Generates an answer grounded in the retrieved documentation.
5. Includes references to the original document chunks.

The project is being developed incrementally, from basic document ingestion to more advanced retrieval techniques.

## 2. Project roadmap

The following roadmap describes the progression of the project and its current implementation.

| Stage | Component | Implementation |
|---|---|---|
| 1 | Document ingestion | Collect and read Markdown documents |
| 2 | Text preprocessing | Clean Kubernetes documentation |
| 3 | Document chunking | Split documents using the model tokenizer |
| 4 | Embeddings | Generate vectors using Sentence Transformers |
| 5 | Vector storage | Store chunks and embeddings in JSONL |
| 6 | Retrieval | Find the most similar chunks using cosine similarity |
| 7 | Answer generation | Send retrieved context to an OpenAI language model |
| 8 | Retrieval improvements | Offset mapping, improved chunk boundaries, evaluation |
| 9 | Advanced RAG | Hybrid search, reranking, and vector databases |

Stages 1–7 represent the current implementation. The remaining stages are planned improvements.

---

## 3. Current architecture

The application has two main processes: document ingestion and question processing.

### Document ingestion

```text
Kubernetes Markdown documents
            |
            v
       Text cleanup
            |
            v
        Tokenizer
            |
            v
      Document chunks
            |
            v
     Local embedding model
      all-MiniLM-L6-v2
            |
            v
       JSONL storage
```

### Question processing

```text
       User question
            |
            v
      Query embedding
            |
            v
     Similarity search
            |
            v
    Top 3 matching chunks
            |
            v
       OpenAI API
            |
            v
    Answer with citations
```

## 4. Implementation details

### Stage 1 — Document ingestion

**Objective:** Collect documents that will form the knowledge base.

I'm using the public Kubernetes documentation:

https://github.com/kubernetes/website/tree/main/content/en/docs

I've started with selected Markdown documents rather than ingesting the entire repository.

The ingestion script recursively searches the local `docs/` directory for Markdown files.

Each document receives an identifier based on its relative file path. This makes it easier to trace retrieved chunks back to their original documents.

**Learning notes:**
- Reading files using Python's `pathlib`.
- Searching nested directories using `rglob()`.
- Working with Markdown files.
- Creating document identifiers.

**Status:** Implemented.

### Stage 2 — Document cleaning

**Objective:** Remove website-specific formatting while retaining the meaningful content.

Kubernetes documentation uses Hugo shortcodes, Markdown links, and HTML comments. These can introduce unnecessary text into the embedding process.

The `content_cleanup()` function uses Python regular expressions to clean the documentation.

Current cleanup rules:

- Preserve the `text` attribute from `glossary_tooltip` shortcodes.
- Remove HTML comments (`<!-- ... -->`).
- Remove Hugo shortcode wrapper tags (`{{< ... >}}`).
- Preserve Markdown link text while removing the associated URL.

**Learning notes:**
- Python regular expressions.
- `re.sub()` and capture groups.
- Preserving useful content while removing formatting instructions.
- Why document preprocessing matters for retrieval.

**Status:** Implemented. Additional cleanup rules may be introduced as new Markdown patterns are encountered.

### Stage 3 — Document chunking

**Objective:** Divide long documents into smaller sections that the embedding model can process.

The embedding model used in this project, `all-MiniLM-L6-v2`, has a maximum input sequence length of 256 tokens.

Passing an entire document that exceeds this limit to `model.encode()` results in truncation. Text beyond the model's input limit is not represented in the resulting embedding.

#### Initial implementation: Word-based chunking

I initially used Python's regular expressions to split documents based on whitespace-separated words.

The `re.finditer(r"\S+", content)` expression identifies each sequence of non-whitespace characters and records its position in the original text.

These positions are then used to extract chunks from the original document.

The initial implementation also introduced overlapping chunks.

#### Problem encountered

Words and tokens are not the same.

The embedding model uses WordPiece tokenization, meaning a single word can be split into multiple tokens.

For example:

```text
Kubernetes
    |
    v
ku + ##ber + ##net + ##es
```

A chunk containing 150 whitespace-separated words could therefore contain significantly more than 256 model tokens.

This meant that word-based chunking could not reliably respect the embedding model's input limit.

#### Current implementation: Token-based chunking

I changed the implementation to use the tokenizer belonging to `all-MiniLM-L6-v2`.

The tokenizer splits the original document into windows based on its actual token count.

Current configuration:

| Setting | Value |
|---|---:|
| Embedding model | all-MiniLM-L6-v2 |
| Maximum sequence length | 256 tokens |
| Chunk size | 251 tokens |
| Overlap | 50 tokens |

The tokenizer uses `return_overflowing_tokens=True` to ensure that the remaining document is returned in additional windows instead of being discarded.

The token IDs are decoded back into text before generating embeddings.

The decoded text is tokenized again to verify that each chunk fits within the model's maximum sequence length.

**Learning notes:**
- Difference between words and tokens.
- Tokenizers and WordPiece tokenization.
- Model input-length restrictions.
- Token-based chunking.
- Overflow handling.
- Overlapping chunks and context preservation.

**Status:** Implemented. Exploring offset mapping to improve how original text is extracted.

### Stage 4 — Embedding generation

**Objective:** Convert each text chunk into a numerical representation that can be compared with other text.

An embedding represents the meaning of a piece of text as a vector of numbers.

I'm using the local Sentence Transformers model:

`sentence-transformers/all-MiniLM-L6-v2`

The model produces a 384-dimensional vector for each chunk.

```python
embeddings = model.encode(
    chunks_text,
    normalize_embeddings=True
)
```

The `normalize_embeddings=True` parameter ensures that each vector has a length of 1.

This allows me to use the dot product to calculate cosine similarity during retrieval.

The embedding model runs locally, so no external embedding API call is required.

**Learning notes:**
- What embeddings represent.
- Sentence Transformers.
- NumPy arrays.
- Vector dimensions.
- Vector normalization.
- Why the same embedding model must be used for document chunks and user queries.

**Status:** Implemented.

### Stage 5 — Vector storage

**Objective:** Persist the document chunks and their embeddings so that they can be searched later.

Each chunk is stored as an individual JSONL record.

The record contains:

| Field | Purpose |
|---|---|
| `doc_id` | Identifies the original document |
| `chunk_id` | Uniquely identifies the individual chunk |
| `source` | Original document path |
| `text` | Actual text of the chunk |
| `embedding` | Numerical vector generated by the embedding model |

Example:

```json
{
  "doc_id": "concepts/workloads/pods",
  "chunk_id": "concepts/workloads/pods#chunk-1",
  "source": "docs/concepts/workloads/pods.md",
  "text": "A Pod is the smallest deployable unit...",
  "embedding": [0.12, 0.43, 0.87]
}
```

The actual embedding contains 384 numbers.

The output is stored in:

`vectors/vectors.jsonl`

JSONL was chosen initially because it is simple to inspect and doesn't require a separate database.

**Learning notes:**
- JSON and JSONL.
- Document metadata.
- Unique document and chunk identifiers.
- Persisting embeddings.
- Retrieving original text using chunk identifiers.

**Status:** Implemented.

### Stage 6 — Similarity search and retrieval

**Objective:** Given a question, find the document chunks that are most similar in meaning.

When a user submits a question, the same embedding model converts the question into a numerical vector.

The query vector is compared against all stored document vectors.

```python
scores = vectors @ query_vector
```

Because the vectors are normalized, the dot product is equivalent to cosine similarity.

The resulting scores are sorted to identify the three most similar chunks.

```python
k_matches = np.argsort(scores)[::-1][:3]
```

The matching indexes are used to retrieve the original text and document metadata from the vector store.

**Learning notes:**
- Vector similarity.
- Dot products and matrix multiplication.
- Cosine similarity.
- NumPy array dimensions.
- `argsort()` and ranking.
- Top-K retrieval.

**Status:** Implemented.

### Stage 7 — Answer generation using an LLM

**Objective:** Use the retrieved document chunks to generate an answer to the original question.

The application sends two things to the OpenAI API:

1. The user's original question.
2. The three document chunks retrieved during the search.

The language model receives instructions to:

- Answer using only the supplied documents.
- Cite the relevant chunk IDs and source documents.
- Avoid unsupported information.
- Say that it doesn't know when the supplied documents don't contain enough information.

The application uses OpenAI's Responses API.

**Learning notes:**
- OpenAI API integration.
- Prompt construction.
- Context injection.
- Grounded answer generation.
- Source citations.
- The relationship between retrieval quality and answer quality.

**Status:** Implemented.

---

## 5. Engineering decisions and implementation history

This section documents the problems encountered, the approaches tried, and the reasoning behind implementation changes.

Each significant decision follows the same structure:

- **Problem:** What wasn't working or what limitation did I encounter?
- **Initial approach:** How did I solve it originally?
- **New approach:** What did I change?
- **Why:** Why is the new approach better?
- **Trade-offs:** What did I gain or lose?
- **Validation:** How did I test whether the change worked?

Detailed explanations and code comparisons are maintained in `docs/decisions/`.

### Decision 001 — Moving from word-based to token-based chunking

**Problem**

The original chunking logic split documents using whitespace-separated words.

However, the embedding model uses WordPiece tokenization. A single word can become several tokens.

This caused some chunks to exceed the model's 256-token input limit, even when their word count appeared relatively small.

**Initial approach**

Used Python's `re.finditer(r"\S+", content)` to identify word positions and extract chunks based on a fixed word count.

The implementation also supported overlapping chunks.

**New approach**

Changed the implementation to use the embedding model's own tokenizer.

The tokenizer now splits the entire document into windows using:

- `max_length`
- `stride`
- `return_overflowing_tokens=True`

Each window is then decoded back into text before embedding.

**Why**

Chunking based on the model's own tokens ensures that the chunks are aligned with the actual input units understood by the embedding model.

The overflow setting also ensures that the remaining document is returned in additional chunks instead of being discarded.

**Trade-offs**

- The chunking implementation is more dependent on the selected embedding model's tokenizer.
- The token IDs must be decoded back into text before they can be passed to `model.encode()`.
- Decoding can change capitalization, whitespace, and token boundaries.
- The decoded text must therefore be tokenized again to verify that it still fits the model's input limit.

**Validation**

Tested the implementation using Kubernetes documentation containing technical terms and identifiers.

Compared whitespace-based word counts against actual tokenizer counts and confirmed that word-based chunk sizes could exceed the 256-token limit.

Added a validation step that checks the token count of every decoded chunk before embedding.

### Decision 002 — Introducing overlapping chunks

**Problem**

When a document is divided into separate chunks, information that belongs together can end up on opposite sides of a chunk boundary.

A chunk may therefore lose context that was present in the preceding section.

**Initial approach**

Created independent chunks without repeating any content between consecutive chunks.

**New approach**

Introduced a 50-token overlap between consecutive chunks.

For example, with a five-token chunk size and one-token overlap:

```text
Chunk 1: 0 1 2 3 4
Chunk 2:         4 5 6 7 8
Chunk 3:                 8 9 10 11
```

The tokenizer's `stride` parameter controls this overlap.

**Why**

Repeating a small portion of the previous chunk helps preserve context when information crosses a chunk boundary.

**Trade-offs**

- Some text is stored and embedded more than once.
- The number of chunks increases.
- Retrieval may return overlapping chunks containing similar information.
- Excessive overlap can increase storage and retrieval costs without necessarily improving answer quality.

**Validation**

Inspected consecutive chunks to confirm that the intended overlap was present.

Further retrieval-quality testing is needed to determine whether 50 tokens is the most appropriate overlap for this documentation collection.

### Decision 003 — Using tokenizer offset mapping for chunk extraction

**Status:** Planned improvement.

**Problem**

The current implementation tokenizes the original document, decodes each token window into text, and then tokenizes the decoded text again.

This introduces two problems:

1. The decoded text can have different capitalization and whitespace from the original document because the tokenizer is uncased.
2. Decoding and re-tokenizing can produce different token boundaries, potentially causing a decoded chunk to exceed the model's input limit.

A five-token safety margin was introduced to reduce the likelihood of exceeding the limit, but this is an approximation rather than a guaranteed solution.

**Initial approach**

1. Tokenize the entire document.
2. Split it into token windows.
3. Decode each window into text.
4. Tokenize the decoded text again.
5. Validate the token count.
6. Generate embeddings.

**New approach**

Use `return_offsets_mapping=True` when tokenizing.

The tokenizer provides the starting and ending character positions of each token in the original document.

Instead of decoding token IDs, use these positions to extract the corresponding text directly from the original string.

**Why**

This avoids the decode-and-re-tokenize round trip and preserves the original document's capitalization and whitespace.

It also allows chunk extraction to be based on the actual character positions associated with the selected tokens.

**Trade-offs**

- Token boundaries can occur in the middle of words.
- Chunk boundaries may need to be adjusted to preserve complete words.
- The extracted original text should still be re-tokenized and validated before embedding because its token boundaries can change when processed as an independent string.

**Validation**

To be completed after implementation.

The new approach will be compared against the existing decoding approach using:
- Original versus extracted text.
- Token counts after extraction.
- Preservation of capitalization and whitespace.
- Successful embedding without truncation.
- Retrieval quality.

### Decision 004 — Cleaning Kubernetes documentation before embedding

**Problem**

The original Markdown files contain website-specific formatting, including Hugo shortcodes, HTML comments, and URLs.

These elements are not always useful for semantic retrieval and can introduce unnecessary tokens.

**Initial approach**

Read the Markdown files and pass their contents directly into the chunking and embedding process.

**New approach**

Introduced a `content_cleanup()` function using regular expressions.

The function:
- Preserves glossary terms from `glossary_tooltip`.
- Removes HTML comments.
- Removes Hugo shortcode wrapper tags.
- Preserves Markdown link text while removing URLs.

**Why**

The objective is to retain the meaningful documentation while removing website-rendering instructions and unnecessary formatting.

**Trade-offs**

- Regular expressions are designed around the specific patterns currently encountered.
- More complex