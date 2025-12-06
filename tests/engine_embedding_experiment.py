
import sys
import numpy as np
from pathlib import Path

# Add src to python path
# Add project root to python path
sys.path.append(str(Path(__file__).parent.parent))

from src.embeddings.engine import AssetSearchEngine

def run_experiment():
    print("--- Starting Experiment ---")
    
    # Initialize Engine
    try:
        engine = AssetSearchEngine()
        print(f"Engine initialized on device: {engine.device}")
    except Exception as e:
        print(f"Failed to initialize engine: {e}")
        return

    from src.config import settings

    # Image Paths using Config
    images_dir = settings.DATA_DIR / "images"
    # Using existing files in the directory
    dog_img_path = str(images_dir / "dog0.webp")
    car_img_path = str(images_dir / "car0.jpeg")

    print(f"Dog Image: {dog_img_path}")
    print(f"Car Image: {car_img_path}")

    # 1. Get Image Embeddings
    print("\nComputing image embeddings...")
    dog_vec = np.array(engine.get_image_embedding(dog_img_path))
    car_vec = np.array(engine.get_image_embedding(car_img_path))
    
    print(f"Dog vector shape: {dog_vec.shape}")
    print(f"Car vector shape: {car_vec.shape}")

    # 2. Get Text Embedding
    print("\nComputing text embedding for 'a dog'...")
    text_vec = np.array(engine.get_text_embedding("a dog"))
    print(f"Text vector shape: {text_vec.shape}")

    # 3. Calculate Cosine Similarity
    # Since vectors are already normalized by the engine, dot product is enough
    # Cosine Similarity = (A . B) / (||A|| * ||B||)
    # But ||A|| and ||B|| should be 1.0 (checking just in case)
    
    dog_norm = np.linalg.norm(dog_vec)
    car_norm = np.linalg.norm(car_vec)
    text_norm = np.linalg.norm(text_vec)
    
    print(f"\nVector Norms (should be ~1.0):")
    print(f"Dog: {dog_norm:.4f}")
    print(f"Car: {car_norm:.4f}")
    print(f"Text: {text_norm:.4f}")

    sim_dog = np.dot(dog_vec, text_vec)
    sim_car = np.dot(car_vec, text_vec)

    print(f"\n--- Results ---")
    print(f"Similarity ('a dog' <-> dog image): {sim_dog:.4f}")
    print(f"Similarity ('a dog' <-> car image): {sim_car:.4f}")

    if sim_dog > sim_car:
        print("\n✅ SUCCESS: Dog image is more similar to 'a dog' than car image.")
    else:
        print("\n❌ FAILURE: Something is wrong. Car image score is higher!")

if __name__ == "__main__":
    run_experiment()
