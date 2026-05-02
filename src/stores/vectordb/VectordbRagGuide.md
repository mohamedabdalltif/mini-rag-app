# 🧠 Vector Databases for RAG — The Complete Guide

> A deep-dive reference covering everything you need to know about choosing, configuring, and using vector databases in Retrieval-Augmented Generation (RAG) pipelines.

---

## 📚 Table of Contents

1. [What is RAG and Why Does Storage Matter?](#1-what-is-rag-and-why-does-storage-matter)
2. [Traditional Databases vs Vector Databases](#2-traditional-databases-vs-vector-databases)
3. [How Vector Databases Work Internally](#3-how-vector-databases-work-internally)
4. [Embedding Models — The Bridge Between Text and Vectors](#4-embedding-models--the-bridge-between-text-and-vectors)
5. [Index Types — Speed vs Accuracy Tradeoff](#5-index-types--speed-vs-accuracy-tradeoff)
6. [Distance Metrics — How Similarity is Measured](#6-distance-metrics--how-similarity-is-measured)
7. [Vector Database Comparison](#7-vector-database-comparison)
   - [Pinecone](#71-pinecone)
   - [Qdrant](#72-qdrant)
   - [Weaviate](#73-weaviate)
   - [Chroma](#74-chroma)
   - [Milvus](#75-milvus)
   - [pgvector](#76-pgvector)
   - [FAISS](#77-faiss)
8. [How to Choose the Right Vector DB](#8-how-to-choose-the-right-vector-db)
9. [Chunking Strategy — Often More Important Than the DB](#9-chunking-strategy--often-more-important-than-the-db)
10. [Metadata Filtering — The Hidden Power Feature](#10-metadata-filtering--the-hidden-power-feature)
11. [Hybrid Search — Combining Vector + Keyword Search](#11-hybrid-search--combining-vector--keyword-search)
12. [Production Considerations](#12-production-considerations)
13. [Code Examples](#13-code-examples)
14. [Decision Flowchart](#14-decision-flowchart)
15. [Quick Reference Cheat Sheet](#15-quick-reference-cheat-sheet)

---

## 1. What is RAG and Why Does Storage Matter?

**Retrieval-Augmented Generation (RAG)** is an architecture that enhances LLMs by giving them access to an external knowledge base at inference time. Instead of relying purely on what the model learned during training, RAG systems:

1. **Retrieve** relevant documents from a knowledge store based on the user's query
2. **Augment** the prompt with those documents as context
3. **Generate** an answer using both the query and the retrieved context

```
User Query
    │
    ▼
Embedding Model  ──────────────────────────────────────────┐
    │                                                       │
    ▼                                                       ▼
Query Vector  ──► Vector DB (ANN Search) ──► Top-K Chunks  │
                                                   │        │
                                                   ▼        │
                                           LLM Prompt ◄─────┘
                                                   │
                                                   ▼
                                            Final Answer
```

### Why the vector database choice matters

The vector DB is the **heart of RAG**. Poor choices here cause:

- **Slow retrieval** — users experience long response times
- **Bad recall** — the right chunks are not found, so the LLM hallucinates
- **Filtering failures** — documents from wrong users, time periods, or categories are returned
- **Scaling nightmares** — your DB can't handle production traffic
- **High costs** — over-engineered solutions for simple use cases

Getting this choice right early saves enormous pain later.

---

## 2. Traditional Databases vs Vector Databases

### The fundamental difference

Traditional databases store and retrieve **exact structured data**. Vector databases store and retrieve **meaning**.

| Dimension | Relational DB (SQL) | Document DB (NoSQL) | Vector DB |
|---|---|---|---|
| **Data model** | Tables, rows, columns | JSON documents | High-dimensional float arrays (embeddings) |
| **Query type** | Exact match, range, join | Field filter, key lookup | Approximate nearest neighbor (ANN) |
| **Search semantic** | "Find rows WHERE x = y" | "Find docs WHERE field = value" | "Find items most similar in meaning to this query" |
| **Schema** | Rigid, predefined | Flexible | Fixed vector dimension, flexible metadata |
| **Indexing** | B-Tree, Hash, GiST | Hash, compound | HNSW, IVF, LSH, Flat |
| **Scaling** | Vertical + horizontal (complex) | Horizontal (easier) | Horizontal (built for it) |
| **Consistency** | ACID transactions | Eventually consistent | Eventually consistent (usually) |
| **Use case** | Banking, ERP, CRM | Content, catalogs, user data | Semantic search, RAG, recommendations |

### Why you can't use a regular DB for RAG

Suppose a user asks: *"What are the company's policies on parental leave?"*

The document in your DB is titled: *"Employee Benefits and Family Support Guidelines, Section 4."*

- **SQL query** `WHERE title LIKE '%parental leave%'` → **no results** (different words)
- **Full-text search** `WHERE document @@ 'parental leave'` → **maybe** (keyword-dependent)
- **Vector search** → **finds it** because the meaning of "parental leave" and "family support guidelines" are semantically close in embedding space

This is the core reason: **semantic similarity beats exact string matching** for natural language retrieval.

### Types of traditional databases and their RAG fit

**Relational (PostgreSQL, MySQL, SQLite)**
- Excellent for structured data and business logic
- Can become RAG-capable with `pgvector` extension
- Best when you already have Postgres infrastructure

**Document stores (MongoDB, Firestore, CouchDB)**
- Good for semi-structured content
- Some (like MongoDB Atlas) now include vector search
- Acceptable for small-scale RAG with simple filtering

**Key-Value stores (Redis, DynamoDB, Memcached)**
- Used for caching vector results, not searching
- Redis has a vector module (`RediSearch`) for basic ANN
- Not recommended as primary vector store

**Graph databases (Neo4j, ArangoDB)**
- Excellent for knowledge graphs and reasoning over relationships
- Can complement vector DBs in Graph RAG architectures
- Not a replacement — they answer different questions

**Search engines (Elasticsearch, OpenSearch, Solr)**
- Strong keyword (BM25) search
- Now include vector/ANN support
- Great for hybrid search scenarios

---

## 3. How Vector Databases Work Internally

Understanding what happens under the hood helps you configure and debug your system.

### Step 1 — Ingestion pipeline

```
Raw Document
    │
    ├─► Text Extraction (PDF, HTML, DOCX, etc.)
    │
    ├─► Chunking (split into segments, e.g. 512 tokens each)
    │
    ├─► Embedding Model (e.g. text-embedding-3-small)
    │       Input: "Employees are entitled to 12 weeks parental leave"
    │       Output: [0.023, -0.145, 0.872, ..., 0.031]  ← 1536 floats
    │
    └─► Insert into Vector DB
            - Vector: [0.023, -0.145, ...]
            - Metadata: { source: "hr_policy.pdf", page: 4, date: "2024-01" }
            - ID: "chunk_00421"
```

### Step 2 — Query pipeline

```
User Query: "How long is maternity leave?"
    │
    ├─► Same Embedding Model
    │       Output: [0.019, -0.138, 0.851, ..., 0.028]
    │
    ├─► ANN Search in Vector DB
    │       Find top-5 vectors closest to query vector
    │
    ├─► Optional: Filter by metadata
    │       (e.g., only docs from 2023 onwards)
    │
    └─► Return chunk texts + scores
            Chunk 1 (score 0.94): "Employees are entitled to 12 weeks..."
            Chunk 2 (score 0.87): "Parental benefits include..."
```

### The index — what makes it fast

Without an index, finding the nearest vector requires comparing the query to **every single stored vector** — an O(n) operation. With 10 million chunks, that's painfully slow. Vector DB indexes (HNSW, IVF, etc.) reduce this to approximately O(log n) by building smart data structures that skip most comparisons. This is called **Approximate** Nearest Neighbor because you trade a tiny bit of accuracy for enormous speed gains.

---

## 4. Embedding Models — The Bridge Between Text and Vectors

The embedding model is **just as important** as the vector DB. A bad embedding model will produce terrible retrieval regardless of how good your DB is.

### Key properties of embedding models

| Property | What it means | Impact |
|---|---|---|
| **Dimension** | Number of floats in each vector | Higher = more expressive but more storage/compute |
| **Max tokens** | Maximum input length | Affects chunking strategy |
| **Context awareness** | Does it capture sentence vs document context? | Affects retrieval quality |
| **Multilingual** | Does it work across languages? | Important for non-English content |
| **Symmetric vs asymmetric** | Same model for query and docs, or different? | Affects how you set up the pipeline |

### Popular embedding models

| Model | Dimensions | Max tokens | Notes |
|---|---|---|---|
| `text-embedding-3-small` (OpenAI) | 1536 | 8191 | Fast, cheap, great quality |
| `text-embedding-3-large` (OpenAI) | 3072 | 8191 | Best OpenAI quality, more expensive |
| `text-embedding-ada-002` (OpenAI) | 1536 | 8191 | Older, still widely used |
| `all-MiniLM-L6-v2` (HuggingFace) | 384 | 256 | Fast, free, good for short texts |
| `bge-large-en-v1.5` (BAAI) | 1024 | 512 | Excellent open-source quality |
| `e5-large-v2` | 1024 | 512 | Strong multilingual variant available |
| `cohere-embed-v3` | 1024 | 512 | Good with `input_type` specification |
| `jina-embeddings-v2` | 768 | 8192 | Long context, open source |

### Critical rule — use the same model for indexing and querying

The query vector and document vectors **must** be produced by the same model. Mixing models produces random garbage results.

---

## 5. Index Types — Speed vs Accuracy Tradeoff

The index is the internal data structure that makes vector search fast. Choosing the right one significantly impacts performance.

### HNSW — Hierarchical Navigable Small World

**The recommended default for most RAG systems.**

HNSW builds a multi-layer graph where each vector is connected to its nearest neighbors. At query time, the algorithm starts at the top layer (long-range connections, fast traversal) and progressively refines the search in lower layers (short-range, precise). It's conceptually similar to how highway systems work — you zoom out to get close, then zoom in to find the exact destination.

**Key tuning parameters:**

| Parameter | What it controls | Default | Notes |
|---|---|---|---|
| `m` | Number of edges per node | 16 | Higher = better recall, more memory |
| `ef_construction` | Search depth during build | 200 | Higher = better quality index, slower build |
| `ef` (query time) | Search depth at query | 100 | Higher = better recall, slower query |

**Pros:** Best recall-to-speed ratio, works well with dynamic inserts, supported by all major vector DBs

**Cons:** High memory usage (stores the graph structure), slower index build time

**Use when:** You have sufficient RAM and want the best query performance. This is the right choice for >90% of RAG deployments.

---

### IVF — Inverted File Index

IVF clusters vectors into `n_list` buckets using k-means. At query time, it searches only the closest `n_probe` clusters instead of all vectors.

**Key parameters:**

| Parameter | What it controls |
|---|---|
| `n_list` | Number of clusters (typical: sqrt(n_vectors)) |
| `n_probe` | How many clusters to search (higher = better recall, slower) |

**Pros:** Much lower memory usage than HNSW, faster to build

**Cons:** Lower recall than HNSW, sensitive to cluster quality, doesn't handle dynamic inserts well

**Use when:** Memory is constrained and you have a mostly static dataset.

---

### IVF-PQ — IVF with Product Quantization

Extends IVF by compressing each vector using Product Quantization — splitting the vector into subvectors and encoding each with a small codebook. A 1536-dim float32 vector (~6KB) can be compressed to ~96 bytes with minimal recall loss.

**Use when:** You have billions of vectors and simply cannot store raw float32 vectors in RAM.

---

### Flat — Brute Force

No approximation — compares the query to every single stored vector. Guarantees 100% recall.

**Use when:** Dataset is small (<100k vectors) and you need exact results. Also useful as a baseline to measure recall of other indexes.

---

### LSH — Locality Sensitive Hashing

Hashes similar vectors to the same bucket using random projections. Historically significant but now largely obsolete — HNSW outperforms it on every metric.

**Use when:** Almost never in modern RAG systems.

---

### Summary table

| Index | Recall | Query Speed | Build Speed | Memory | Handles Inserts |
|---|---|---|---|---|---|
| Flat | 100% | ❌ Slow | ✅ Instant | Low | ✅ Yes |
| HNSW | ~95–99% | ✅✅ Very fast | ⚠️ Slow | ❌ High | ✅ Yes |
| IVF | ~85–95% | ✅ Fast | ✅ Medium | ✅ Low | ❌ Poor |
| IVF-PQ | ~75–90% | ✅ Fast | ✅ Medium | ✅✅ Very low | ❌ Poor |
| LSH | ~70–85% | ✅ Fast | ✅ Fast | ✅ Low | ✅ Yes |

---

## 6. Distance Metrics — How Similarity is Measured

The distance metric defines how "closeness" between two vectors is computed.

### Cosine Similarity

Measures the **angle** between two vectors, ignoring their magnitude. Two vectors pointing in the same direction = similarity of 1.0. Opposite directions = -1.0.

```
cosine_similarity(A, B) = (A · B) / (|A| × |B|)
```

**Best for:** Text embeddings. Most embedding models are trained to produce vectors where cosine similarity corresponds to semantic similarity. **Use this by default for RAG.**

---

### Dot Product (Inner Product)

The raw product of two vectors without normalization. Equivalent to cosine similarity when vectors are normalized to unit length.

```
dot_product(A, B) = A · B = Σ(Aᵢ × Bᵢ)
```

**Best for:** When embedding models produce normalized vectors (most do), dot product is slightly faster than cosine. Pinecone and many others use this internally.

---

### Euclidean Distance (L2)

Measures the **straight-line distance** between two points in the vector space.

```
L2(A, B) = sqrt(Σ(Aᵢ - Bᵢ)²)
```

**Best for:** Image embeddings, cases where magnitude carries meaning. Generally not ideal for text RAG.

---

### Manhattan Distance (L1)

Sum of absolute differences across all dimensions. Rarely used in RAG.

**Practical rule:** For text RAG, always start with **cosine similarity**. Check your embedding model's documentation — many models specify which metric they were optimized for.

---

## 7. Vector Database Comparison

### 7.1 Pinecone

**Type:** Fully managed SaaS cloud service

**Architecture:** Serverless pods, proprietary internals, REST/gRPC API

**Best for:** Teams that want zero infrastructure management and are willing to pay for it.

**Strengths:**
- Zero ops — no servers, no configuration, just an API key
- Scales automatically, handles billions of vectors
- Fast out of the box
- Good SDKs for Python, Node, Go, Java
- Namespaces for multi-tenancy

**Weaknesses:**
- Vendor lock-in (proprietary format, hard to migrate)
- Expensive at scale ($70+/month for production pods)
- Metadata filtering limited on serverless tier
- No self-hosting option
- Closed source — you can't inspect internals

**Pricing model:** Serverless (pay per read/write unit) or pod-based (fixed monthly cost per pod type)

**When to choose Pinecone:**
- Your team has no DevOps capacity
- You need to ship fast and cost isn't the primary concern
- You're building a startup MVP or POC

**Quick start:**
```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")
index = pc.Index("my-rag-index")

# Upsert vectors
index.upsert(vectors=[
    {"id": "chunk_001", "values": [0.1, 0.2, ...], "metadata": {"source": "hr.pdf"}},
])

# Query
results = index.query(vector=[0.1, 0.2, ...], top_k=5, include_metadata=True)
```

---

### 7.2 Qdrant

**Type:** Open-source, self-hostable, also available as managed cloud

**Architecture:** Written in Rust, single binary deployment, gRPC + REST API

**Best for:** Production RAG systems that need rich filtering and prefer self-hosting.

**Strengths:**
- Extremely rich payload (metadata) filtering — filter by any combination of fields alongside vector search
- HNSW index with great performance
- Written in Rust → very fast and memory efficient
- Supports named vectors (store multiple embeddings per document — e.g. title vector + body vector)
- Payload indexing for fast filter pre-filtering
- Supports sparse vectors for hybrid search (BM42)
- Excellent documentation
- Can run locally with a single Docker command

**Weaknesses:**
- Newer ecosystem than Pinecone or Weaviate
- Managed cloud is smaller than Pinecone's
- Less community content and tutorials

**When to choose Qdrant:**
- You need complex metadata filtering (e.g., "only docs from user X, category Y, after date Z")
- You want open-source + self-host control
- You care about performance and memory efficiency
- You want sparse + dense hybrid search

**Quick start:**
```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient("localhost", port=6333)

client.create_collection(
    collection_name="rag_docs",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

client.upsert(
    collection_name="rag_docs",
    points=[
        PointStruct(
            id=1,
            vector=[0.1, 0.2, ...],
            payload={"source": "hr.pdf", "date": "2024-01", "user_id": "u_123"}
        ),
    ]
)

# Search with filter
from qdrant_client.models import Filter, FieldCondition, MatchValue

results = client.search(
    collection_name="rag_docs",
    query_vector=[0.1, 0.2, ...],
    query_filter=Filter(
        must=[FieldCondition(key="user_id", match=MatchValue(value="u_123"))]
    ),
    limit=5,
)
```

---

### 7.3 Weaviate

**Type:** Open-source, self-hostable, managed cloud available

**Architecture:** Written in Go, GraphQL + REST API, modular vectorizer plugins

**Best for:** Multi-modal RAG, teams that want built-in vectorization, GraphQL lovers.

**Strengths:**
- Can vectorize content automatically using built-in modules (OpenAI, Cohere, HuggingFace modules)
- Multi-modal support (text + images in same DB)
- Native hybrid search (BM25 + vector)
- Class-based schema with typed properties
- Strong GraphQL query interface
- Good integrations with LangChain and LlamaIndex

**Weaknesses:**
- More complex configuration and schema setup
- Heavier resource footprint than Qdrant
- GraphQL learning curve for teams unfamiliar with it
- Slower startup and initialization vs Qdrant

**When to choose Weaviate:**
- Multi-modal data (images + text)
- You want the DB to handle vectorization automatically
- You prefer GraphQL queries
- You need LangChain/LlamaIndex integration out of the box

**Quick start:**
```python
import weaviate

client = weaviate.Client("http://localhost:8080")

# Create schema
client.schema.create_class({
    "class": "Document",
    "vectorizer": "text2vec-openai",
    "properties": [
        {"name": "content", "dataType": ["text"]},
        {"name": "source", "dataType": ["string"]},
    ]
})

# Add object (vectorized automatically)
client.data_object.create(
    data_object={"content": "Employees get 12 weeks parental leave", "source": "hr.pdf"},
    class_name="Document"
)

# Query
result = client.query.get("Document", ["content", "source"]) \
    .with_near_text({"concepts": ["parental leave policy"]}) \
    .with_limit(5) \
    .do()
```

---

### 7.4 Chroma

**Type:** Open-source, in-process or client-server, Python-first

**Architecture:** Written in Python (core), SQLite or DuckDB backend, simple HTTP API

**Best for:** Local development, prototyping, learning RAG fundamentals.

**Strengths:**
- Simplest possible setup — runs in-process, no Docker needed
- Great for notebooks and experimentation
- First-class LangChain integration
- Built-in embedding functions (wraps OpenAI, HuggingFace, etc.)
- Very intuitive Python API

**Weaknesses:**
- Not production-ready at scale (performance degrades with millions of vectors)
- Limited metadata filtering capability
- No HNSW tuning options exposed
- Not built for multi-user, high-concurrency production traffic
- Persistent storage can be fragile

**When to choose Chroma:**
- You're learning RAG
- Local development and prototyping
- Demos, hackathons, academic projects
- Small internal tools (<500k vectors, low traffic)

**Quick start:**
```python
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./chroma_db")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="your-api-key",
    model_name="text-embedding-3-small"
)

collection = client.get_or_create_collection(
    name="rag_docs",
    embedding_function=openai_ef
)

# Add documents (automatically embeds)
collection.add(
    documents=["Employees are entitled to 12 weeks parental leave"],
    metadatas=[{"source": "hr.pdf"}],
    ids=["chunk_001"]
)

# Query
results = collection.query(
    query_texts=["How long is maternity leave?"],
    n_results=5
)
```

---

### 7.5 Milvus

**Type:** Open-source, distributed, enterprise-grade, managed as Zilliz Cloud

**Architecture:** Written in Go + C++, distributed microservices (etcd, MinIO, Pulsar), gRPC API

**Best for:** Billion-scale vector search, enterprise deployments with strict performance SLAs.

**Strengths:**
- Designed from the ground up for massive scale
- Multiple index types per collection (HNSW, IVF, DISKANN, etc.)
- GPU acceleration support
- Supports sparse + dense vectors
- Strong consistency guarantees
- Rich SDK (Python, Java, Go, Node, C++)
- Zilliz Cloud as managed option

**Weaknesses:**
- Very complex infrastructure to self-host (requires etcd, MinIO, message queue)
- Overkill for most small-to-medium RAG systems
- Steeper learning curve
- Local dev is painful compared to Chroma or Qdrant

**When to choose Milvus:**
- You're dealing with 100M–10B+ vectors
- Enterprise with dedicated infrastructure team
- You need GPU-accelerated search
- Compliance requires self-hosted on-prem

---

### 7.6 pgvector

**Type:** PostgreSQL extension — adds vector column type and ANN search

**Architecture:** Runs entirely inside your existing PostgreSQL instance

**Best for:** Teams already running Postgres who want to add semantic search without a new service.

**Strengths:**
- No new service to deploy, monitor, or pay for
- Full SQL power — join vector search with structured data in one query
- ACID transactions — vector inserts are part of your regular transactions
- Great for multi-tenant SaaS (row-level security applies to vectors too)
- Works with any Postgres hosting (AWS RDS, Supabase, Neon, self-hosted)
- Supports HNSW and IVF indexes (added in recent versions)

**Weaknesses:**
- Slower ANN performance than dedicated vector DBs (especially at >5M vectors)
- HNSW index is memory-intensive within Postgres shared memory
- Concurrent heavy write + read workloads cause Postgres lock contention
- Not ideal when vector search is the only or dominant workload

**When to choose pgvector:**
- You're already running Postgres (especially on Supabase or Neon)
- You need to JOIN vector results with relational data
- Your scale is under ~5–10M vectors
- You want the simplest possible operational model

**Quick start:**
```sql
-- Install extension
CREATE EXTENSION vector;

-- Create table with vector column
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    source TEXT,
    created_at TIMESTAMP,
    embedding vector(1536)
);

-- Create HNSW index
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);

-- Insert with vector
INSERT INTO documents (content, source, embedding)
VALUES ('Employees get 12 weeks parental leave', 'hr.pdf', '[0.1, 0.2, ...]');

-- Semantic search with filter
SELECT content, source, 1 - (embedding <=> '[0.1, 0.2, ...]') AS similarity
FROM documents
WHERE created_at > '2023-01-01'
ORDER BY embedding <=> '[0.1, 0.2, ...]'
LIMIT 5;
```

---

### 7.7 FAISS

**Type:** In-memory library (Facebook AI Research), not a database

**Architecture:** C++ library with Python bindings, no server, no persistence by default

**Best for:** Research, custom pipelines, maximum control over the search algorithm.

**Strengths:**
- Fastest raw ANN performance available
- Widest variety of index types (IVF, PQ, HNSW, LSH, Flat, and combinations)
- GPU support via FAISS-GPU
- Used internally by many vector DBs
- Full control over every parameter

**Weaknesses:**
- Not a database — no server, no REST API, no persistence out of the box
- No metadata filtering whatsoever
- You must handle serialization, storage, and serving yourself
- Not suitable for production without significant custom engineering

**When to choose FAISS:**
- ML research and experiments
- You're building a custom vector DB or search engine
- You need GPU-accelerated billion-scale search in a custom system
- Benchmark comparisons

---

## 8. How to Choose the Right Vector DB

### Decision framework

Answer these questions in order:

**Question 1 — What is your vector count?**
- Under 1M vectors → Chroma (dev), pgvector, Qdrant
- 1M – 50M vectors → Qdrant, Weaviate, Pinecone
- 50M – 500M vectors → Pinecone, Weaviate, Milvus
- 500M+ vectors → Milvus, Pinecone enterprise

**Question 2 — Do you need metadata filtering?**
- No filtering needed → Any option works
- Simple filtering (1–2 fields) → All modern vector DBs
- Complex multi-field filtering → Qdrant (best), pgvector (SQL power), Weaviate
- Per-user isolation / multi-tenancy → pgvector (row-level security), Qdrant (payload index), Pinecone (namespaces)

**Question 3 — Who manages infrastructure?**
- No ops team, pay for convenience → Pinecone
- DevOps team available, prefer open-source → Qdrant or Weaviate
- Already running Postgres → pgvector (zero new infra)
- Enterprise, need on-prem → Milvus

**Question 4 — What stage are you at?**
- Prototype / learning → Chroma
- Early production → Qdrant or pgvector
- Scaling production → Qdrant, Weaviate, Pinecone
- Enterprise / massive scale → Milvus

**Question 5 — Do you need hybrid search?**
- Yes (vector + keyword BM25) → Weaviate (native), Qdrant (sparse vectors + dense), Elasticsearch

**Question 6 — Multi-modal data?**
- Yes (text + images) → Weaviate
- Text only → Any of the above

### Opinionated recommendations

| Scenario | Recommended DB | Reason |
|---|---|---|
| Learning RAG for the first time | Chroma | Simplest API, least setup |
| Production app, small team, no ops | Pinecone | Zero ops, fast start |
| Production app, want open-source | Qdrant | Best performance + filtering |
| Already on Postgres / Supabase | pgvector | No new service needed |
| Billion-scale enterprise | Milvus | Built for this exact scale |
| Multi-modal (text + images) | Weaviate | Native multi-modal support |
| Research / custom pipeline | FAISS | Maximum control |
| Hybrid search critical | Weaviate or Qdrant | Both support dense + sparse |

---

## 9. Chunking Strategy — Often More Important Than the DB

Even the best vector DB returns bad results if chunks are poorly designed. Chunking is the process of splitting your source documents into segments for embedding.

### Fixed-size chunking

Split by token count, ignoring content structure.

```python
from langchain.text_splitter import TokenTextSplitter

splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = splitter.split_text(document_text)
```

**Pros:** Simple, predictable  
**Cons:** Cuts sentences mid-thought, loses context at boundaries

**Overlap:** Always use 10–15% overlap between chunks (e.g., 50 tokens overlap for 512-token chunks). This ensures that information at chunk boundaries is captured in at least one chunk.

---

### Semantic chunking

Split based on semantic shifts — when the topic changes, start a new chunk.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

splitter = SemanticChunker(OpenAIEmbeddings(), breakpoint_threshold_type="percentile")
chunks = splitter.split_text(document_text)
```

**Pros:** Chunks are semantically coherent, better retrieval quality  
**Cons:** Slower (requires embeddings during chunking), variable chunk sizes

---

### Recursive character text splitter

Split by paragraphs → sentences → words, respecting natural language boundaries.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""]
)
```

**Pros:** Respects document structure, good default for most cases  
**Cons:** Not as precise as semantic chunking

---

### Hierarchical / parent-child chunking

Store large parent chunks + small child chunks. Embed and retrieve small chunks (more precise), but return large chunks to the LLM (more context).

```
Parent chunk (2000 tokens) → stored for LLM context
    ├── Child chunk 1 (200 tokens) → embedded for retrieval
    ├── Child chunk 2 (200 tokens) → embedded for retrieval
    └── Child chunk 3 (200 tokens) → embedded for retrieval
```

**Best for:** Long documents, cases where you need precise retrieval + rich context.

---

### Chunk size recommendations

| Content type | Recommended chunk size | Overlap |
|---|---|---|
| Short Q&A / FAQ | 128–256 tokens | 20 tokens |
| General documents | 512–1024 tokens | 50–100 tokens |
| Legal / technical docs | 1024–2048 tokens | 100–200 tokens |
| Code | Per function/class | None |

---

## 10. Metadata Filtering — The Hidden Power Feature

Metadata filtering is one of the most underrated aspects of production RAG. Almost every real system needs it.

### Why it matters

Without filtering, your RAG system might return:
- Documents from other users in a multi-tenant app (security issue)
- Outdated documents that have been superseded (accuracy issue)
- Documents from the wrong product, category, or region (relevance issue)

### What to store as metadata

```python
metadata = {
    # Source tracking
    "source_file": "hr_policy_v3.pdf",
    "page_number": 4,
    "section": "Benefits",
    
    # Temporal
    "created_at": "2024-01-15",
    "updated_at": "2024-03-01",
    "version": 3,
    
    # Access control
    "user_id": "u_abc123",
    "organization_id": "org_xyz",
    "access_level": "confidential",
    
    # Content classification
    "category": "hr",
    "language": "en",
    "document_type": "policy",
    
    # Quality signals
    "chunk_index": 4,
    "total_chunks": 12,
}
```

### Filtering patterns

**Exact match:**
```python
# Qdrant
Filter(must=[FieldCondition(key="category", match=MatchValue(value="hr"))])

# pgvector (SQL)
WHERE category = 'hr'
```

**Range filter:**
```python
# Qdrant
Filter(must=[FieldCondition(key="created_at", range=Range(gte="2023-01-01"))])

# pgvector
WHERE created_at >= '2023-01-01'
```

**Multi-tenant isolation:**
```python
# Always include user/org filter in production
Filter(must=[
    FieldCondition(key="organization_id", match=MatchValue(value=current_org_id)),
    FieldCondition(key="access_level", match=MatchAny(any=user_access_levels))
])
```

**Important:** Index your metadata fields for performance. Most vector DBs require explicit payload index creation:

```python
# Qdrant — create payload index
client.create_payload_index(
    collection_name="rag_docs",
    field_name="user_id",
    field_schema="keyword"
)
```

---

## 11. Hybrid Search — Combining Vector + Keyword Search

Pure vector search can miss exact terminology, product codes, names, and technical jargon. Hybrid search combines:

- **Dense search** (vector / semantic): finds conceptually related content
- **Sparse search** (BM25 / keyword): finds exact term matches

### When to use hybrid search

- Documents contain product codes, IDs, or technical terms that must match exactly
- User queries are short (1–3 words) — pure vector search struggles
- You need to balance precision (exact match) and recall (semantic match)

### Reciprocal Rank Fusion (RRF)

The standard approach for combining results from two search systems:

```python
def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    scores = {}
    for rank, doc in enumerate(dense_results):
        scores[doc.id] = scores.get(doc.id, 0) + 1 / (k + rank + 1)
    for rank, doc in enumerate(sparse_results):
        scores[doc.id] = scores.get(doc.id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### Implementations

**Weaviate** — built-in hybrid search:
```python
result = client.query.get("Document", ["content"]) \
    .with_hybrid(query="parental leave policy", alpha=0.5) \
    .with_limit(5) \
    .do()
# alpha=0: pure BM25 | alpha=1: pure vector | alpha=0.5: balanced
```

**Qdrant** — sparse + dense vectors:
```python
client.search_batch(
    collection_name="docs",
    requests=[
        SearchRequest(vector=NamedVector(name="dense", vector=dense_vec), limit=20),
        SearchRequest(vector=NamedSparseVector(name="sparse", vector=sparse_vec), limit=20),
    ]
)
# Then fuse results with RRF
```

**Elasticsearch / OpenSearch** — mature hybrid with BM25 + KNN:
```python
query = {
    "query": {"match": {"content": "parental leave"}},
    "knn": {"field": "embedding", "query_vector": [...], "k": 10, "num_candidates": 100}
}
```

---

## 12. Production Considerations

### Latency targets

| Operation | Target | Notes |
|---|---|---|
| Embedding generation | < 100ms | Use batching, cached embeddings |
| Vector search (ANN) | < 20ms | HNSW with good tuning |
| End-to-end retrieval | < 150ms | Embedding + search + metadata lookup |
| Total RAG response | < 3s | Including LLM generation |

### Reliability patterns

**Connection pooling:**
```python
# Qdrant — use async client for high concurrency
from qdrant_client import AsyncQdrantClient

client = AsyncQdrantClient("localhost", port=6333)
```

**Retry logic:**
```python
import tenacity

@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(min=1, max=10),
    retry=tenacity.retry_if_exception_type(Exception)
)
async def search_with_retry(query_vector, filters):
    return await client.search(...)
```

**Caching frequent queries:**
```python
import hashlib, json
from functools import lru_cache

def cache_key(query: str, filters: dict) -> str:
    return hashlib.md5(f"{query}{json.dumps(filters, sort_keys=True)}".encode()).hexdigest()
```

### Monitoring metrics to track

| Metric | Why it matters |
|---|---|
| Retrieval latency (p50, p95, p99) | User experience |
| Recall@K | Are the right chunks being retrieved? |
| Vector DB memory usage | HNSW indexes can be large |
| Embedding cache hit rate | Reduces embedding API costs |
| Query volume per collection | Capacity planning |
| Index freshness | How stale is the knowledge base? |

### Index update strategies

**Batch ingestion (offline):**
```
New Documents → Queue → Nightly Batch Embedding → Bulk Upsert → Index Rebuild
```
Use when: Documents update daily/weekly and slight staleness is acceptable.

**Real-time ingestion:**
```
New Document → Webhook → Embed → Upsert immediately
```
Use when: Freshness is critical (news, support tickets, live data).

**Versioned indexes:**
```
Active Index: v2 (serving traffic)
Shadow Index: v3 (being built)
→ Cutover when v3 is ready
```
Use when: Major schema changes or full re-indexing is needed.

---

## 13. Code Examples

### Full RAG pipeline (Python + Qdrant + OpenAI)

```python
import openai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import uuid

# ── Setup ────────────────────────────────────────────────────────────────────
openai_client = openai.OpenAI(api_key="your-openai-key")
qdrant = QdrantClient("localhost", port=6333)
COLLECTION = "rag_knowledge_base"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

qdrant.recreate_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
)

# ── Embed function ────────────────────────────────────────────────────────────
def embed(texts: list[str]) -> list[list[float]]:
    response = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]

# ── Ingest documents ──────────────────────────────────────────────────────────
def ingest(documents: list[dict]):
    """documents: list of {"text": str, "metadata": dict}"""
    texts = [d["text"] for d in documents]
    vectors = embed(texts)
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"text": doc["text"], **doc["metadata"]}
        )
        for doc, vector in zip(documents, vectors)
    ]
    qdrant.upsert(collection_name=COLLECTION, points=points)
    print(f"Ingested {len(points)} chunks")

# ── Retrieve ──────────────────────────────────────────────────────────────────
def retrieve(query: str, top_k: int = 5, filters: dict = None) -> list[dict]:
    query_vector = embed([query])[0]
    qdrant_filter = None
    if filters:
        qdrant_filter = Filter(must=[
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filters.items()
        ])
    results = qdrant.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        query_filter=qdrant_filter,
        limit=top_k,
        with_payload=True
    )
    return [{"text": r.payload["text"], "score": r.score, **r.payload} for r in results]

# ── Generate answer ───────────────────────────────────────────────────────────
def answer(query: str, top_k: int = 5, filters: dict = None) -> str:
    chunks = retrieve(query, top_k=top_k, filters=filters)
    context = "\n\n---\n\n".join(c["text"] for c in chunks)
    
    system_prompt = """You are a helpful assistant. Answer the user's question using ONLY the provided context.
If the context doesn't contain enough information, say so clearly. Do not make up information."""
    
    user_prompt = f"""Context:
{context}

Question: {query}

Answer:"""
    
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content

# ── Example usage ─────────────────────────────────────────────────────────────
ingest([
    {"text": "Employees are entitled to 12 weeks of paid parental leave.", "metadata": {"source": "hr_policy.pdf", "category": "hr"}},
    {"text": "The annual performance review cycle runs from January to March.", "metadata": {"source": "hr_policy.pdf", "category": "hr"}},
])

print(answer("How long is parental leave?", filters={"category": "hr"}))
```

---

## 14. Decision Flowchart

```
START
  │
  ▼
Are you prototyping / learning?
  ├─ YES → Chroma (local, zero setup)
  └─ NO
       │
       ▼
     Do you already use PostgreSQL?
       ├─ YES → pgvector (add extension, no new service)
       └─ NO
            │
            ▼
          Do you have an ops team?
            ├─ NO → Pinecone (fully managed)
            └─ YES
                 │
                 ▼
               Scale > 500M vectors?
                 ├─ YES → Milvus
                 └─ NO
                      │
                      ▼
                    Need multi-modal (images + text)?
                      ├─ YES → Weaviate
                      └─ NO → Qdrant (best for most production RAG)
```

---

## 15. Quick Reference Cheat Sheet

### Vector DB at a glance

| | Pinecone | Qdrant | Weaviate | Chroma | Milvus | pgvector |
|---|---|---|---|---|---|---|
| **Hosting** | Managed only | Self/Cloud | Self/Cloud | Self | Self/Cloud | Self |
| **Open source** | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Setup difficulty** | ⭐ Easy | ⭐⭐ Medium | ⭐⭐⭐ Hard | ⭐ Easiest | ⭐⭐⭐⭐ Hard | ⭐ Easy |
| **Scale** | Billions | Very large | Large | Small–Med | Billions | Medium |
| **Filtering** | ✅ | ✅✅✅ | ✅✅ | ✅ | ✅✅ | ✅✅✅ (SQL) |
| **Hybrid search** | ❌ | ✅ | ✅ | ❌ | ✅ | ⚠️ |
| **Multi-modal** | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| **Cost** | 💰💰💰 | Free/💰 | Free/💰 | Free | Free/💰 | Free |

### Index type cheat sheet

| Index | Recall | Speed | Memory | Use case |
|---|---|---|---|---|
| **HNSW** | ~97% | ⚡⚡⚡ | High | Default — use this |
| **IVF** | ~90% | ⚡⚡ | Low | Memory-constrained |
| **IVF-PQ** | ~80% | ⚡⚡ | Very low | Billions of vectors |
| **Flat** | 100% | ⚡ | Low | <100k vectors, exact needed |

### Distance metric cheat sheet

| Metric | Use when |
|---|---|
| **Cosine** | Text embeddings (default for RAG) |
| **Dot product** | Normalized vectors (slightly faster than cosine) |
| **Euclidean (L2)** | Image embeddings, magnitude matters |

### Chunk size cheat sheet

| Content type | Chunk size | Overlap |
|---|---|---|
| FAQ / short answers | 128–256 tokens | 20 |
| General documents | 512–768 tokens | 50–75 |
| Long-form / legal | 1024–2048 tokens | 100–200 |
| Code | Function/class boundary | 0 |

---

*Last updated: 2025. Maintained for production RAG engineers and ML practitioners.*