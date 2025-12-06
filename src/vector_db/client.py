from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.embeddings.engine import AssetSearchEngine
from src.config import settings
import os
import glob
from pathlib import Path


class VectorDB:
    """Vector database client for indexing and searching image assets using CLIP embeddings."""
    
    def __init__(self, collection_name="visual_assets"):
        self.collection_name = collection_name
        
        # Connect to Qdrant using config settings
        self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        
        # Initialize search engine for embeddings
        self.engine = AssetSearchEngine()
        
        # CLIP ViT-B/32 produces 512-dimensional vectors
        self.vector_size = 512
        
        # Auto-create collection if it doesn't exist
        self._create_collection_if_not_exists()

    def _create_collection_if_not_exists(self):
        """Create collection with proper configuration if it doesn't exist."""
        if not self.client.collection_exists(self.collection_name):
            print(f"Creating collection '{self.collection_name}'...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE  # Optimal for normalized CLIP embeddings
                )
            )
            print(f"Collection '{self.collection_name}' created successfully.")
        else:
            print(f"Collection '{self.collection_name}' already exists.")

    def index_directory(self, dir_path: str):
        """
        Index all images from a directory into the vector database.
        
        Args:
            dir_path: Path to directory containing images (jpg, png)
        """
        dir_path = Path(dir_path)
        if not dir_path.exists():
            print(f"Error: Directory {dir_path} does not exist.")
            return
        
        # Get all image files
        image_paths = list(dir_path.glob("*.jpg")) + \
                      list(dir_path.glob("*.jpeg")) + \
                      list(dir_path.glob("*.png"))
        
        if not image_paths:
            print(f"No images found in {dir_path}")
            return
        
        print(f"\nFound {len(image_paths)} images. Starting indexing...")
        
        points = []
        indexed_count = 0
        error_count = 0
        
        for idx, img_path in enumerate(image_paths):
            try:
                # Generate embedding for the image
                vector = self.engine.get_image_embedding(str(img_path))
                
                if vector is None:
                    error_count += 1
                    continue
                
                # Create point with metadata
                point = models.PointStruct(
                    id=idx,  # In production, use UUID for better uniqueness
                    vector=vector,
                    payload={
                        "path": str(img_path.absolute()),
                        "filename": img_path.name,
                        "extension": img_path.suffix
                    }
                )
                points.append(point)
                
                # Batch upload every 10 images for efficiency
                if len(points) >= 10:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    indexed_count += len(points)
                    points = []
                    print(f"Indexed {indexed_count}/{len(image_paths)} images...")
                    
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
                error_count += 1
        
        # Upload remaining points
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            indexed_count += len(points)
        
        print(f"\n✓ Indexing complete!")
        print(f"  Successfully indexed: {indexed_count}")
        print(f"  Errors: {error_count}")

    def search(self, text_query: str, limit: int = 3):
        """
        Search for images using a text description.
        
        Args:
            text_query: Natural language description of desired image
            limit: Maximum number of results to return
            
        Returns:
            List of dicts with 'path', 'filename', and 'score' keys
        """
        # Convert text query to embedding
        text_vector = self.engine.get_text_embedding(text_query)
        
        # Search for nearest neighbors in vector space
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=text_vector,
            limit=limit
        )
        
        # Format results
        results = []
        for hit in search_result.points:
            results.append({
                "path": hit.payload["path"],
                "filename": hit.payload["filename"],
                "score": hit.score
            })
        
        return results

    def get_collection_info(self):
        """Get information about the current collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return None


# --- TEST ---
if __name__ == "__main__":
    import os
    
    db = VectorDB()
    
    # Path to DVC dataset
    data_path = os.path.join("data", "raw", "images")
    
    # 1. Index images from DVC dataset (run once)
    if os.path.exists(data_path):
        print(f"\n📁 Indexing images from: {data_path}")
        db.index_directory(data_path)
    else:
        print(f"❌ Folder {data_path} not found! Make sure you've pulled the dataset with DVC.")
        print("Run: dvc pull")
    
    # 2. Search examples
    print("\n" + "="*70)
    print("🔍 Search Examples")
    print("="*70)
    
    test_queries = [
        "a person riding a bike",
        "a dog playing in the park",
        "a car on the street"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 70)
        results = db.search(query, limit=3)
        
        for i, res in enumerate(results, 1):
            print(f"{i}. Score: {res['score']:.4f} | File: {res['filename']}")
    
    # 3. Collection stats
    print("\n" + "="*70)
    print("📊 Collection Statistics")
    print("="*70)
    info = db.get_collection_info()
    if info:
        print(f"Collection: {info['name']}")
        print(f"Points indexed: {info['points_count']}")
        print(f"Status: {info['status']}")

