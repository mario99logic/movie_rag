import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration class for RAG system"""

    # OpenAI Configuration.
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = "text-embedding-3-small"

    # Chunking Configuration.
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

    # Retrieval Configuration.
    TOP_K = int(os.getenv("TOP_K", 3))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.4))

    # Vector Database Configuration.
    CHROMA_DB_PATH = "./chroma_db"
    COLLECTION_NAME = "rag_collection"

    # Dataset Configuration.
    DATA_PATH = "./data"

    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        return True


config = Config()
