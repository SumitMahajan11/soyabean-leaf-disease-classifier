import torch
import numpy as np
from PIL import Image
from pathlib import Path
import logging
from typing import Dict, Tuple, Any, Optional
import pickle
from torchvision import transforms
from production_inference_enhanced import SoybeanDiseaseClassifierEnhanced

logger = logging.getLogger(__name__)

class SoybeanSimilarityVerifier:
    """
    Robust soybean crop identification mechanism that determines whether an uploaded leaf image 
    is similar to the soybean dataset used in training.
    """
    
    def __init__(self, classifier, embedding_bank_path: Path, 
                 high_threshold: float = 0.78, 
                 low_threshold: float = 0.62,
                 device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load classifier and extract feature extractor
        self.feature_extractor = self._create_feature_extractor(classifier)
        
        # Load reference embeddings
        self.embeddings = None
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        
        if embedding_bank_path.exists():
            try:
                self.embeddings = np.load(embedding_bank_path)
                logger.info(f"Soybean reference embeddings loaded: shape {self.embeddings.shape}")
            except Exception as e:
                logger.error(f"Failed to load reference embeddings: {e}")
        else:
            logger.warning(f"Reference embeddings not found at {embedding_bank_path}")
    
    def _create_feature_extractor(self, classifier):
        """
        Create a feature extractor by removing the final classification layer
        """
        # Extract the backbone (EfficientNet-B3) without the final classifier
        backbone = classifier.primary_model.features  # Features up to the final conv layer
        
        # Create a feature extractor module
        class FeatureExtractor(torch.nn.Module):
            def __init__(self, features, device):
                super().__init__()
                self.features = features
                self.device = device
                
            def forward(self, x):
                x = self.features(x)
                x = torch.nn.functional.adaptive_avg_pool2d(x, 1)  # Global average pooling
                x = x.view(x.size(0), -1)  # Flatten
                return x
        
        feature_extractor = FeatureExtractor(backbone, self.device).to(self.device)
        feature_extractor.eval()
        return feature_extractor
    
    @staticmethod
    def _create_feature_extractor_static(classifier):
        """
        Static method to create a feature extractor by removing the final classification layer
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Extract the backbone (EfficientNet-B3) without the final classifier
        backbone = classifier.primary_model.features  # Features up to the final conv layer
        
        # Create a feature extractor module
        class FeatureExtractor(torch.nn.Module):
            def __init__(self, features, device):
                super().__init__()
                self.features = features
                self.device = device
                
            def forward(self, x):
                x = self.features(x)
                x = torch.nn.functional.adaptive_avg_pool2d(x, 1)  # Global average pooling
                x = x.view(x.size(0), -1)  # Flatten
                return x
        
        feature_extractor = FeatureExtractor(backbone, device).to(device)
        feature_extractor.eval()
        return feature_extractor
    
    def _extract_embedding(self, image: Image.Image) -> torch.Tensor:
        """
        Extract normalized embedding from an image
        """
        # Define the same preprocessing used during training
        preprocess = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
        
        # Preprocess image
        input_tensor = preprocess(image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
            # L2 normalize the features
            features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        return features
    
    def verify(self, image: Image.Image, image_resolution: Tuple[int, int]) -> Dict[str, Any]:
        """
        Verify if the image is similar to soybean dataset
        """
        if self.embeddings is None or self.embeddings.size == 0:
            return {
                "decision": "uncertain",
                "similarity_score": None,
                "high_threshold": self.high_threshold,
                "low_threshold": self.low_threshold,
                "image_resolution": image_resolution,
                "warning": "Soybean reference embeddings not available; skipping similarity check.",
                "status": "uncertain"
            }
        
        # Extract embedding from the input image
        emb = self._extract_embedding(image)
        emb_np = emb.detach().cpu().numpy().astype(np.float32)
        
        # Compute cosine similarity with all reference embeddings (vectorized)
        # Cosine similarity for L2 normalized vectors is just dot product
        similarities = np.dot(self.embeddings, emb_np[0])
        max_similarity = float(similarities.max().item())
        
        # Apply decision logic based on thresholds
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


def build_soybean_reference_embeddings(images_dir: Path, classifier, output_path: Path):
    """
    Offline script to build soybean reference embeddings from training dataset
    """
    logger.info(f"Building soybean reference embeddings from {images_dir}")
    
    # Create feature extractor
    feature_extractor = SoybeanSimilarityVerifier._create_feature_extractor_static(classifier)
    
    # Define preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Find all image files in the directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(images_dir.rglob(f"*{ext}"))
        image_paths.extend(images_dir.rglob(f"*{ext.upper()}"))
    
    logger.info(f"Found {len(image_paths)} images to process")
    
    embeddings = []
    
    for i, img_path in enumerate(image_paths):
        try:
            image = Image.open(img_path).convert('RGB')
            input_tensor = preprocess(image).unsqueeze(0).to(feature_extractor.device)
            
            with torch.no_grad():
                features = feature_extractor(input_tensor)
                # L2 normalize the features
                features = torch.nn.functional.normalize(features, p=2, dim=1)
            
            embeddings.append(features.squeeze(0).cpu().numpy())
            
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(image_paths)} images")
                
        except Exception as e:
            logger.warning(f"Failed to process {img_path}: {e}")
            continue
    
    if embeddings:
        # Stack all embeddings
        embeddings_array = np.stack(embeddings, axis=0).astype(np.float32)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        np.save(output_path, embeddings_array)
        logger.info(f"Saved reference embeddings to {output_path} with shape {embeddings_array.shape}")
    else:
        logger.error("No embeddings were generated!")