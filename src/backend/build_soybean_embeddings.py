"""
Script to build soybean reference embeddings from the training dataset
"""
import sys
from pathlib import Path

# Add the src directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.soybean_similarity_verifier import build_soybean_reference_embeddings
from code.production_inference_enhanced import SoybeanDiseaseClassifierEnhanced
from backend.config import Config

def main():
    print("Building soybean reference embeddings...")
    
    # Load the classifier
    classifier = SoybeanDiseaseClassifierEnhanced(model_dir=str(Config.ENHANCED_MODEL_DIR))
    
    # Define paths
    # You'll need to point this to your soybean dataset directory
    soybean_images_dir = Path(input("Enter path to soybean images directory: "))
    
    if not soybean_images_dir.exists():
        print(f"Directory {soybean_images_dir} does not exist!")
        return
    
    output_path = Config.ENHANCED_MODEL_DIR / "soybean_reference_embeddings.npy"
    
    print(f"Building embeddings from {soybean_images_dir}")
    print(f"Output path: {output_path}")
    
    build_soybean_reference_embeddings(
        images_dir=soybean_images_dir,
        classifier=classifier,
        output_path=output_path
    )
    
    print("Embedding building complete!")

if __name__ == "__main__":
    main()