import sys
from pathlib import Path

# Add src to python path if needed (though local import should work if run as module)
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import settings

def main():
    print(" Smart Asset Search Initialized ")
    print(f"Configuration loaded:")
    print(f" - Model: {settings.MODEL_NAME}")
    print(f" - Device: {settings.DEVICE}")
    print(f" - Qdrant: {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")

    # Initialize Engine
    try:
        from src.embeddings.engine import AssetSearchEngine
        print("\nInitializing Search Engine...")
        engine = AssetSearchEngine()
        print("Search Engine loaded successfully.")
    except Exception as e:
        print(f"\nWarning: Could not load Search Engine: {e}")

if __name__ == "__main__":
    main()
