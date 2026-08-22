"""
Soybean Embedding-Based Verifier
Uses EfficientNet-B4 feature embeddings to verify if an image matches the soybean dataset.
"""
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
import logging
from typing import Dict, Tuple, Any
from torchvision import transforms

logger = logging.getLogger(__name__)


class EfficientNetFeatureExtractor(nn.Module):
    """Extract features from EfficientNet-B4 penultimate layer"""
    
    def __init__(self, efficientnet_model):
        super().__init__()
        # EfficientNet-B4 features and avgpool
        self.features = efficientnet_model.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class SoybeanEmbeddingVerifier:
    """
    Verify if an uploaded image matches soybean dataset using cosine similarity.
    """
    
    def __init__(self, classifier, embedding_bank_path: Path,
                 high_threshold: float = 0.70,
                 low_threshold: float = 0.55,
                 device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        
        # Create feature extractor
        self.feature_extractor = EfficientNetFeatureExtractor(classifier.primary_model).to(self.device)
        self.feature_extractor.eval()
        
        # Load reference embeddings
        self.embeddings = None
        if embedding_bank_path.exists():
            try:
                self.embeddings = np.load(embedding_bank_path)
                logger.info(f"Soybean reference embeddings loaded: shape {self.embeddings.shape}")
            except Exception as e:
                logger.error(f"Failed to load reference embeddings: {e}")
        else:
            logger.warning(f"Soybean embedding bank not found at {embedding_bank_path}. Verification gate will be permissive.")
        
        # Define preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _extract_embedding(self, image: Image.Image) -> torch.Tensor:
        """Extract L2-normalized embedding from image"""
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
            # L2 normalize
            features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        return features
    
    def verify(self, image: Image.Image, image_resolution: Tuple[int, int]) -> Dict[str, Any]:
        """
        Verify if image matches soybean dataset
        
        Returns:
            dict with keys: decision, similarity_score, status, warning
        """
        if self.embeddings is None or self.embeddings.size == 0:
            return {
                "decision": "uncertain",
                "similarity_score": None,
                "high_threshold": self.high_threshold,
                "low_threshold": self.low_threshold,
                "image_resolution": image_resolution,
                "warning": "Soybean embedding bank is empty or missing. Skipping verification gate.",
                "status": "uncertain"
            }
        
        # Extract embedding
        emb = self._extract_embedding(image)
        emb_np = emb.detach().cpu().numpy().astype(np.float32)
        
        # Compute cosine similarity with all reference embeddings (vectorized)
        similarities = np.dot(self.embeddings, emb_np[0])
        max_similarity = float(similarities.max().item())
        
        # Apply decision logic
        if max_similarity >= self.high_threshold:
            decision = "soybean"
            status = "verified"
            warning = None
        elif max_similarity >= self.low_threshold:
            decision = "likely_soybean"
            status = "warning"
            warning = "Leaf partially matches soybean patterns. Results may be unreliable."
        else:
            decision = "not_soybean"
            status = "rejected"
            warning = None
        
        logger.info(
            "Soybean similarity verification: score=%.4f, decision=%s, resolution=%s, status=%s",
            max_similarity, decision, image_resolution, status
        )
        
        return {
            "decision": decision,
            "similarity_score": max_similarity,
            "high_threshold": self.high_threshold,
            "low_threshold": self.low_threshold,
            "image_resolution": image_resolution,
            "warning": warning,
            "status": status
        }
