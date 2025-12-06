from qdrant_client import QdrantClient
from src.config import settings

def get_client() -> QdrantClient:
    """Returns a QdrantClient instance."""
    return QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
