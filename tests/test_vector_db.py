#!/usr/bin/env python3
"""
Test script to demonstrate Qdrant vector database functionality.
This script will:
1. Index images from the images/ directory
2. Perform several search queries
3. Display results
"""

import sys
from pathlib import Path

# Add project root to path (parent of tests directory)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_db.client import VectorDB


def main():
    print("\n" + "="*70)
    print(" 🔍 Smart Asset Search - Vector Database Test (DVC Dataset)")
    print("="*70 + "\n")
    
    # Initialize VectorDB
    print("Initializing Vector Database...")
    db = VectorDB()
    print("✓ Vector Database initialized.\n")
    
    # Step 1: Index images from DVC dataset
    print("STEP 1: Indexing images from DVC dataset")
    print("-" * 70)
    
    import os
    data_path = os.path.join("data", "raw", "images")
    
    if os.path.exists(data_path):
        db.index_directory(data_path)
    else:
        print(f"❌ Error: {data_path} not found!")
        print("Make sure you've pulled the dataset with: dvc pull")
        return
    
    # Step 2: Show collection info
    print("\n" + "="*70)
    print("STEP 2: Collection Information")
    print("-" * 70)
    info = db.get_collection_info()
    if info:
        print(f"Collection: {info['name']}")
        print(f"Points indexed: {info['points_count']}")
        print(f"Status: {info['status']}")
    
    # Step 3: Perform search queries
    print("\n" + "="*70)
    print("STEP 3: Search Queries")
    print("="*70)
    
    test_queries = [
        "a person riding a bike",
        "a dog playing in the park",
        "a car on the street",
        "people sitting at a table"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 70)
        
        results = db.search(query, limit=3)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['filename']}")
                print(f"   Score: {result['score']:.4f}")
        else:
            print("No results found.")
    
    print("\n" + "="*70)
    print("✓ Test completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
