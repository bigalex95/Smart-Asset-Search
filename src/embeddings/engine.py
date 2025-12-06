
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from src.config import settings

class AssetSearchEngine:
    def __init__(self, model_name=settings.MODEL_NAME):
        # 1. Get requested device from config
        requested_device = settings.DEVICE
        
        # 2. Check availability
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            print(f"Warning: CUDA requested but not available. Fallback to CPU.")
            self.device = "cpu"
        elif requested_device.startswith("mps") and not torch.backends.mps.is_available():
            print(f"Warning: MPS requested but not available. Fallback to CPU.")
            self.device = "cpu"
        else:
            self.device = requested_device

        print(f"Loading model {model_name} to {self.device}...")
        
        # Load pre-trained model and processor
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def get_image_embedding(self, image_path: str):
        """Converts an image to a vector"""
        try:
            image = Image.open(image_path)
            # Improve: Handle RGBA images by converting to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            
            # Normalize vector
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            return image_features.cpu().numpy().flatten().tolist()
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

    def get_text_embedding(self, text: str):
        """Converts text to a vector"""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features.cpu().numpy().flatten().tolist()

# --- TEST ---
if __name__ == "__main__":
    # Simple test block
    engine = AssetSearchEngine()
    print("Model loaded successfully.")
