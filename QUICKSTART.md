# Quick Start Guide

Follow these steps to get the RAG Retrieval System up and running:

## Step 1: Set up Environment

1. Make sure you're in the project directory:

2. Activate the virtual environment:

   ```bash
   source venv/bin/activate
   ```

3. Create your `.env` file:

   ```bash
   cp .env.example .env
   ```

4. Edit `.env` and add your OpenAI API key:

   ```bash
   # Use your preferred editor.
   nano .env
   ```

   Add your key:

   ```
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

   Retrival parameters can also be set in the `.env` file:

   ```
   CHUNK_SIZE=500
   CHUNK_OVERLAP=50
   TOP_K=3
   SIMILARITY_THRESHOLD=0.5
   ```

## Step 2: Ingest the Data

Run the ingestion script to process the movies dataset:

```bash
python scripts/ingest_data.py --file data/top_movies.csv --column overview
```

You should see:

Loading data from data/top_movies.csv...
Processing 100 documents...
Chunk size: 500, Overlap: 50
Created ~150 chunks
Ingesting into vector database...
Successfully ingested ~150 chunks

## Step 3: Start the Backend

```bash
python backend/app.py

```

The API will be available at http://localhost:8000

You should see:

```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## Step 4: Test the API

In a new terminal, test the health endpoint:

```bash
curl http://localhost:8000/health
```

Test a query:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "prison escape and redemption", "top_k": 3, "similarity_threshold": 0.5}'
```

## Step 5: Open the Frontend

1. Open `frontend/index.html` in your web browser

   Or serve it with a simple HTTP server:

   ```bash
   cd frontend
   python -m http.server 3000
   ```

2. Navigate to http://localhost:3000

3. Try some queries:
   - "prison escape and redemption"
   - "mafia crime family"
   - "artificial intelligence and robots"
   - "time travel science fiction"
