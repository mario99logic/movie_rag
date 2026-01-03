# RAG Retrieval System

A simple Retrieval-Augmented Generation (RAG) system that focuses on the retrieval component. This system ingests text data, creates embeddings, stores them in a vector database, and retrieves relevant context chunks based on user queries.

## Overview

This project demonstrates the core retrieval functionality of a RAG system without implementing LLM generation. It provides a clean, minimal interface for querying a knowledge base and retrieving semantically similar content.

[Project demo video](https://drive.google.com/file/d/1xQKTyHhb34op3ESNkNEmoT92iq375XU7/view?usp=sharing)

## Features

- **Data Ingestion**: Process CSV datasets and chunk text intelligently
- **Vector Storage**: Persistent storage using ChromaDB
- **Semantic Search**: Retrieve relevant chunks using OpenAI embeddings
- **REST API**: FastAPI backend with query endpoints
- **Web UI**: Clean, responsive interface for queries and results
- **Configurable Parameters**: Adjust chunk size, overlap, top-k, and similarity threshold

## Project Structure

```
rag/
├── backend/
│   ├── app.py              # FastAPI application
│   ├── config.py           # Configuration management
│   ├── embeddings.py       # OpenAI embedding service
│   ├── vectordb.py         # ChromaDB vector database
│   └── retriever.py        # Retrieval logic
├── frontend/
│   └── index.html          # Web interface
├── scripts/
│   └── ingest_data.py      # Data ingestion script
├── data/
│   └── top_movies.csv      # Sample dataset (top 100 rated movies)
├── .env.example            # Environment variables template
├── .gitignore              # Git ignore rules
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Design Decisions

### Dataset Selection

**Dataset**: Top 100 Rated Movies from Kaggle based on IMDB

**Source**: [Kaggle Top Rated Movies Dataset](https://www.kaggle.com/datasets/mohsin31202/top-rated-movies-dataset)

**Why this dataset:**

- **Rich Text Content**: Each movie has a 200-400 character plot overview - essential for RAG embeddings
- **Better than alternatives**: Other movie datasets (e.g., IMDB Top 100) only have metadata (genre, ratings) without plot descriptions, making them unsuitable for semantic search
- Real-world use case that's relatable and easy to evaluate
- Enables natural language queries beyond simple keyword matching

**Target Users & Use Cases:**

- Movie enthusiasts looking for films by theme or mood rather than exact title
- Users wanting recommendations based on plot elements, not just genres
- People searching for "movies like X" based on story similarity
- Anyone exploring films through natural conversation rather than filters

**Expected Conversation Patterns:**

- Thematic searches: "prison escape and redemption story"
- Plot-based queries: "artificial intelligence taking over the world"
- Similarity searches: "crime family drama like The Godfather"
- Mood/genre blends: "dark psychological thriller"
- Historical contexts: "world war 2 based on true story"

### Vector Database Selection

**Choice**: ChromaDB

**Justification:**

1. **Easy Setup**: Runs locally without external services, perfect for development
2. **Persistent Storage**: Saves embeddings to disk, no re-embedding on restart
3. **Python Native**: Seamless integration with Python applications
4. **Zero Configuration**: Works out of the box with minimal setup
5. **Cost**: Completely free and open-source
6. **Performance**: Efficient for small to medium datasets with HNSW indexing
7. **Cosine Similarity**: Built-in support for cosine similarity metric

### Embeddings

**Model**: OpenAI `text-embedding-3-small` - High quality embeddings at minimal cost (~$0.02 per 1M tokens, well under $1 budget). Assignment requires OpenAI embeddings, and this model provides the best cost/performance balance.

### Chunking Strategy

**Configuration:**

- **Chunk Size**: 500 characters
- **Overlap**: 50 characters (10% overlap)

**Rationale:**

1. **Chunk Size (500)**:

   - Large enough to maintain context and semantic meaning
   - Small enough for focused, relevant results
   - Optimal for the Q&A format where answers are 150-300 chars
   - Balances between granularity and context preservation

2. **Overlap (50)**:
   - Prevents information loss at chunk boundaries
   - Ensures context continuity across chunks
   - 10% overlap is industry best practice
   - Minimal redundancy while maintaining coherence

**Alternatives Considered:**

- Smaller chunks (200-300): Too granular, loses context
- Larger chunks (1000+): Less precise retrieval
- No overlap: Risk of splitting important information

### Retrieval Parameters

**Top-K**: 3 (default, configurable 1-10)

- Provides enough context without overwhelming the user
- Balances precision and recall
- Typical RAG systems use 3-5 chunks

**Similarity Threshold**: 0.4 (default, configurable 0.0-1.0)

- Filters out low-quality matches
- Cosine similarity 0.4+ indicates reasonable semantic similarity
- User can adjust based on query specificity

**Similarity Metric**: Cosine Similarity

- Standard for text embeddings
- Measures angle between vectors, not magnitude
- Range: 0-1 (higher is more similar)
- Invariant to vector magnitude

### Backend Technology Selection

**Choice**: FastAPI - Modern Python framework with automatic request validation (Pydantic), built-in API docs, and better performance than Flask. Less boilerplate code for the same functionality.

### Frontend Technology Selection

**Choice**: Plain HTML/CSS/JavaScript

**Justification:**

1. **Simplicity**: The UI has minimal functionality - a search box, parameter controls, and results display. No complex state management or routing needed.
2. **Zero Dependencies**: No build tools, no npm packages, no version conflicts. Just open the HTML file and it works.
3. **No Build Step**: Instant development - edit and refresh. No compilation, bundling, or transpiling required.
4. **Reduced Complexity**: React would add unnecessary overhead (React, ReactDOM, bundler, babel, etc.) for a single-page interface.
5. **Better for Assignment**: Fewer dependencies means easier setup for evaluators and lower maintenance burden.
6. **Performance**: Native JS is faster than framework overhead for simple interactions.

## Quick Start

For detailed setup and running instructions, see **[QUICKSTART.md](QUICKSTART.md)**.

**Summary:**

1. Set up `.env` with your OpenAI API key
2. Run ingestion: `python scripts/ingest_data.py --file data/top_movies.csv --column overview`
3. Start backend: `python backend/app.py`
4. Open `frontend/index.html` in your browser

## Usage

### Web Interface

1. Enter your query in the search box
2. Adjust parameters if needed:
   - **Top K**: Number of results to return (1-10)
   - **Similarity Threshold**: Minimum similarity score (0-1)
3. Click "Search" or press Enter
4. View retrieved chunks with similarity scores and metadata

### API Endpoints

**Query Endpoint**

```http
POST /query
Content-Type: application/json

{
  "query": "How do I create a list?",
  "top_k": 3,
  "similarity_threshold": 0.5
}
```

Response:

```json
{
  "query": "How do I create a list?",
  "results": [
    {
      "chunk_id": "doc_6",
      "content": "A list is an ordered mutable collection...",
      "similarity_score": 0.8542,
      "distance": 0.1458,
      "metadata": {...},
      "rank": 1
    }
  ],
  "total_results": 3,
  "parameters": {
    "top_k": 3,
    "similarity_threshold": 0.5
  }
}
```

**Health Check**

```http
GET /health
```

**Collection Info**

```http
GET /collection/info
```

## Example Queries

Try these queries with the Movies dataset:

- "prison escape and redemption"
- "mafia crime family"
- "world war 2 documentary"
- "artificial intelligence and robots"
- "time travel science fiction"
- "superhero saving the world"

## Architecture

### Components

1. **Configuration (`config.py`)**

   - Centralized configuration using environment variables
   - Validation of required settings
   - Default values for all parameters

2. **Embedding Service (`embeddings.py`)**

   - Wraps OpenAI API for creating embeddings
   - Supports single and batch embedding creation
   - Error handling for API failures

3. **Vector Database (`vectordb.py`)**

   - ChromaDB client management
   - Document storage with metadata
   - Similarity search with configurable parameters

4. **Retriever (`retriever.py`)**

   - High-level retrieval interface
   - Similarity threshold filtering
   - Result formatting and ranking

5. **FastAPI Application (`app.py`)**

   - REST API endpoints
   - Request/response validation with Pydantic
   - CORS support for frontend
   - Error handling

6. **Data Ingestion (`ingest_data.py`)**
   - CSV data loading with pandas
   - Configurable text chunking
   - Batch processing and embedding

### Data Flow

```
User Query → FastAPI → Retriever → Vector DB → ChromaDB
                ↓                      ↑
           Embedding Service → OpenAI API
```

1. User submits query via UI
2. Frontend sends POST request to `/query`
3. FastAPI validates request
4. Retriever creates query embedding
5. Vector DB searches for similar chunks
6. Results filtered by similarity threshold
7. Formatted results returned to frontend
8. UI displays chunks with scores

## Configuration

All configuration is managed through environment variables:

```bash
# Required
OPENAI_API_KEY=your_key_here

# Optional (defaults shown)
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=3
SIMILARITY_THRESHOLD=0.5
```
