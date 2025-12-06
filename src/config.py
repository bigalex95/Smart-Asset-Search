import os
from pathlib import Path

class Config:
    # Project Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / os.getenv("DATA_DIR", "data")
    
    # Qdrant
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    
    # Model
    MODEL_NAME = os.getenv("MODEL_NAME", "openai/clip-vit-base-patch32")
    DEVICE = os.getenv("DEVICE", "cuda")

settings = Config()
