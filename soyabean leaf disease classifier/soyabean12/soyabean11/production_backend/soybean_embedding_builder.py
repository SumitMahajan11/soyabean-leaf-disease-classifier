"""
Offline embedding builder for soybean verification gate.

This script builds a reference bank of soybean embeddings 
using the V2 EfficientNet-B4 features.
"""

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# Import local modules
from config import Config
from production_inference_enhanced import SoybeanDiseaseClassifierEnhanced
from soybean_verifier import EfficientNetFeatureExtractor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def build_embedding_bank(images_dir: Path, output_path: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    classifier = SoybeanDiseaseClassifierEnhanced(model_dir=str(Config.ENHANCED_MODEL_DIR))
    feature_extractor = EfficientNetFeatureExtractor(classifier.primary_model).to(device)
    feature_extractor.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    embeddings = []

    image_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp"):
        image_paths.extend(images_dir.rglob(ext))

    if not image_paths:
        logger.warning("No images found in %s", images_dir)

    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = feature_extractor(tensor)
                feat = torch.nn.functional.normalize(feat, dim=1)
            embeddings.append(feat.squeeze(0).cpu().numpy())
        except Exception as e:
            logger.warning("Failed to process %s: %s", img_path, e)

    if not embeddings:
        logger.error("No embeddings generated; aborting.")
        return

    emb_array = np.stack(embeddings, axis=0).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, emb_array)
    logger.info("Saved %d embeddings (dim=%d) to %s", emb_array.shape[0], emb_array.shape[1], output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build soybean embedding bank.")
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing soybean training images.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(
            Config.PROJECT_ROOT
            / "models"
            / "CNN_trained_models"
            / "soybean_embedding_bank.npy"
        ),
        help="Path to output .npy file for embeddings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images_dir)
    output_path = Path(args.output_path)
    build_embedding_bank(images_dir, output_path)


if __name__ == "__main__":
    main()
