import sys
from pathlib import Path

# Add src to python path if needed (though local import should work if run as module)
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import settings
from src.vector_db.client import VectorDB


def print_header():
    """Print application header."""
    print("\n" + "="*60)
    print(" 🔍 Smart Asset Search - Vector Database Demo")
    print("="*60)
    print(f"Configuration:")
    print(f"  Model: {settings.MODEL_NAME}")
    print(f"  Device: {settings.DEVICE}")
    print(f"  Qdrant: {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
    print("="*60 + "\n")


def show_menu():
    """Display main menu."""
    print("\nOptions:")
    print("  1. Index images from directory")
    print("  2. Search for images")
    print("  3. Show collection info")
    print("  4. Exit")
    return input("\nSelect option (1-4): ").strip()


def index_images(db: VectorDB):
    """Handle image indexing."""
    dir_path = input("\nEnter directory path (default: data/raw/images): ").strip()
    if not dir_path:
        dir_path = "data/raw/images"
    
    print(f"\nIndexing images from: {dir_path}")
    db.index_directory(dir_path)


def search_images(db: VectorDB):
    """Handle image search."""
    query = input("\nEnter search query (e.g., 'a red car'): ").strip()
    if not query:
        print("Search query cannot be empty.")
        return
    
    limit = input("Number of results (default: 3): ").strip()
    limit = int(limit) if limit.isdigit() else 3
    
    print(f"\n🔍 Searching for: '{query}'")
    print("-" * 60)
    
    results = db.search(query, limit=limit)
    
    if not results:
        print("No results found. Make sure you have indexed images first.")
        return
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['filename']}")
        print(f"   Score: {result['score']:.4f}")
        print(f"   Path: {result['path']}")


def show_collection_info(db: VectorDB):
    """Display collection statistics."""
    print("\n📊 Collection Information")
    print("-" * 60)
    
    info = db.get_collection_info()
    if info:
        print(f"Name: {info['name']}")
        print(f"Points: {info['points_count']}")
        print(f"Status: {info['status']}")
    else:
        print("Could not retrieve collection information.")


def main():
    """Main application loop."""
    print_header()
    
    # Initialize VectorDB
    try:
        print("Initializing Vector Database...")
        db = VectorDB()
        print("✓ Vector Database initialized successfully.\n")
    except Exception as e:
        print(f"❌ Error initializing Vector Database: {e}")
        return
    
    # Main loop
    while True:
        choice = show_menu()
        
        if choice == "1":
            index_images(db)
        elif choice == "2":
            search_images(db)
        elif choice == "3":
            show_collection_info(db)
        elif choice == "4":
            print("\n👋 Goodbye!\n")
            break
        else:
            print("Invalid option. Please select 1-4.")


if __name__ == "__main__":
    main()
