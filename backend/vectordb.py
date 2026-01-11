import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
from backend.config import Config
from backend.embeddings import EmbeddingService


class VectorDatabase:
    """Vector database using ChromaDB for storing and retrieving embeddings"""

    def __init__(self, config: Config, embedding_service: EmbeddingService):
        self.config = config
        self.embedding_service = embedding_service
        self.client = chromadb.PersistentClient(
            path=config.CHROMA_DB_PATH, settings=Settings(anonymized_telemetry=False)
        )
        self.collection = None

    def create_collection(self, reset: bool = False):
        """
        Create or get collection

        Args:
            reset: If True, delete existing collection and create new one
        """
        if reset:
            try:
                self.client.delete_collection(name=self.config.COLLECTION_NAME)
            except Exception:
                pass

        self.collection = self.client.get_or_create_collection(
            name=self.config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    def add_documents(
        self, documents: List[str], metadatas: List[Dict[str, Any]] = None
    ):
        """
        Add documents to the vector database

        Args:
            documents: List of text chunks to add
            metadatas: Optional list of metadata dictionaries for each document
        """
        if not self.collection:
            self.create_collection()

        if not documents:
            raise ValueError("No documents provided")

        # Generate embeddings for all documents
        embeddings = self.embedding_service.create_embeddings(documents)

        # Generate IDs for documents
        ids = [f"doc_{i}" for i in range(len(documents))]

        # Add to collection
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
            if metadatas
            else [{"index": i} for i in range(len(documents))],
        )

    def query(self, query_text: str, n_results: int = None) -> Dict[str, Any]:
        """
        Query the vector database for similar documents

        Args:
            query_text: Query text
            n_results: Number of results to return (defaults to config.TOP_K)

        Returns:
            Dictionary containing documents, distances, and metadata
        """
        if not self.collection:
            self.create_collection()

        if n_results is None:
            n_results = self.config.TOP_K

        # Generate embedding for query
        query_embedding = self.embedding_service.create_embedding(query_text)

        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=n_results
        )

        return results

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        if not self.collection:
            self.create_collection()

        return {
            "name": self.collection.name,
            "count": self.collection.count(),
            "metadata": self.collection.metadata,
        }
