from openai import (
    OpenAI,
    APIError,
    RateLimitError,
    AuthenticationError,
    APIConnectionError,
)
from typing import List
from backend.config import Config


class EmbeddingService:
    """Service for generating embeddings using OpenAI API.

    Uses text-embedding-3-small model for:
    - High quality semantic embeddings at low cost (~$0.02/1M tokens)
    - Good balance between performance and cost for small datasets
    - Assignment requirement: must use OpenAI embeddings
    """

    def __init__(self, config: Config):
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
        if not text:
            raise ValueError("Text cannot be empty for embedding creation")
        try:
            response = self.client.embeddings.create(input=text, model=self.model)
            return response.data[0].embedding
        except AuthenticationError as e:
            raise AuthenticationError(
                f"OpenAI authentication failed. Check your API key: {str(e)}"
            ) from e
        except RateLimitError as e:
            raise RateLimitError(f"OpenAI rate limit exceeded: {str(e)}") from e
        except APIConnectionError as e:
            raise APIConnectionError(
                f"Failed to connect to OpenAI API: {str(e)}"
            ) from e
        except APIError as e:
            raise APIError(f"OpenAI API error: {str(e)}") from e
        except Exception as e:
            raise Exception(f"Unexpected error creating embedding: {str(e)}") from e

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
        except AuthenticationError as e:
            raise AuthenticationError(
                f"OpenAI authentication failed. Check your API key: {str(e)}"
            ) from e
        except RateLimitError as e:
            raise RateLimitError(f"OpenAI rate limit exceeded: {str(e)}") from e
        except APIConnectionError as e:
            raise APIConnectionError(
                f"Failed to connect to OpenAI API: {str(e)}"
            ) from e
        except APIError as e:
            raise APIError(f"OpenAI API error: {str(e)}") from e
        except Exception as e:
            raise Exception(f"Unexpected error creating embeddings: {str(e)}") from e
