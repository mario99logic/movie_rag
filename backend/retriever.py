from typing import List, Dict, Any
from backend.vectordb import VectorDatabase
from backend.config import Config


class Retriever:
    """Retriever for finding relevant context chunks."""

    def __init__(self, vector_db: VectorDatabase, config: Config):
        self.vector_db = vector_db
        self.top_k = config.TOP_K
        self.similarity_threshold = config.SIMILARITY_THRESHOLD

    def retrieve(
        self, query: str, top_k: int = None, similarity_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context chunks for a query.

        Args:
            query: User query string
            top_k: Number of results to return (overrides config)
            similarity_threshold: Minimum similarity score (overrides config)

        Returns:
            List of dictionaries containing retrieved chunks with scores and metadata
        """
        if not query or not query.strip():
            return []

        k = top_k if top_k is not None else self.top_k
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self.similarity_threshold
        )

        # Query vector database.
        results = self.vector_db.query(query, n_results=k)

        # Process and filter results.
        retrieved_chunks = []

        if results and results.get("documents") and len(results["documents"]) > 0:
            documents = results["documents"][0]
            distances = results["distances"][0]
            metadatas = results.get("metadatas", [[]])[0]
            ids = results.get("ids", [[]])[0]

            for i, (doc, distance, metadata, doc_id) in enumerate(
                zip(documents, distances, metadatas, ids)
            ):
                # Convert distance to similarity score (cosine similarity).
                # ChromaDB returns L2 distance, but we configured it for cosine.
                # For cosine distance: similarity = 1 - distance.
                similarity_score = 1 - distance

                # Filter by similarity threshold
                if similarity_score >= threshold:
                    retrieved_chunks.append(
                        {
                            "chunk_id": doc_id,
                            "content": doc,
                            "similarity_score": round(similarity_score, 4),
                            "distance": round(distance, 4),
                            "metadata": metadata,
                            "rank": i + 1,
                        }
                    )

        return retrieved_chunks
