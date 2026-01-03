from openai import OpenAI
from typing import List
from backend.config import config


class EmbeddingService:
    """Service for generating embeddings using OpenAI API.

    Uses text-embedding-3-small model for:
    - High quality semantic embeddings at low cost (~$0.02/1M tokens)
    - Good balance between performance and cost for small datasets
    - Assignment requirement: must use OpenAI embeddings
    """

    def __init__(self):
        config.validate()
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.EMBEDDING_MODEL  # text-embedding-3-small.

    def create_embedding(self, text: str) -> List[float]:
        """
        Create embedding for a single text string.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        try:
            response = self.client.embeddings.create(input=text, model=self.model)
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"Error creating embedding: {str(e)}")

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            response = self.client.embeddings.create(input=texts, model=self.model)
            return [item.embedding for item in response.data]
        except Exception as e:
            raise Exception(f"Error creating embeddings: {str(e)}")


embedding_service = EmbeddingService()
