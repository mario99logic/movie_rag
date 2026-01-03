"""Backend FastAPI application for RAG retrieval system.
Provides endpoints for querying the vector database and retrieving relevant context chunks."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from backend.retriever import retriever
from backend.vectordb import vector_db
from backend.config import config

app = FastAPI(
    title="RAG Retrieval System",
    description="A simple RAG retrieval system for finding relevant context chunks",
    version="1.0.0",
)

# Add CORS middleware to allow frontend requests.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    query: str = Field(..., min_length=1, description="User query text")
    top_k: Optional[int] = Field(
        None, ge=1, le=10, description="Number of results to return"
    )
    similarity_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Minimum similarity score"
    )


class QueryResponse(BaseModel):
    """Response model for query endpoint."""

    query: str
    results: List[Dict[str, Any]]
    total_results: int
    parameters: Dict[str, Any]


class CollectionInfo(BaseModel):
    """Response model for collection info endpoint."""

    name: str
    count: int
    metadata: Dict[str, Any]


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RAG Retrieval System API",
        "endpoints": {
            "/query": "POST - Query the RAG system",
            "/collection/info": "GET - Get collection information",
            "/health": "GET - Health check",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        info = vector_db.get_collection_info()
        return {"status": "healthy", "collection_count": info.get("count", 0)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.get("/collection/info", response_model=CollectionInfo)
async def get_collection_info():
    """Get information about the vector database collection."""
    try:
        info = vector_db.get_collection_info()
        return info
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting collection info: {str(e)}"
        )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system to retrieve relevant context chunks.

    Args:
        request: QueryRequest with query text and optional parameters

    Returns:
        QueryResponse with retrieved chunks and metadata
    """
    try:
        # Validate collection has data.
        collection_info = vector_db.get_collection_info()
        if collection_info.get("count", 0) == 0:
            raise HTTPException(
                status_code=400,
                detail="Vector database is empty. Please ingest data first.",
            )

        # Perform retrieval.
        results = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
        )

        # Prepare response.
        response = QueryResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            parameters={
                "top_k": request.top_k or config.TOP_K,
                "similarity_threshold": request.similarity_threshold
                or config.SIMILARITY_THRESHOLD,
            },
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
