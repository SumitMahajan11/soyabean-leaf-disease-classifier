"""
Soybean embedding-based crop verification gate using EfficientNet-B3.

This module reuses the production Enhanced EfficientNet model as a feature
extractor and verifies whether an input image looks like the soybean leaves
seen during training, based on cosine similarity in embedding space.

Design:
- Offline script builds an embedding bank from known soybean training images.
- Runtime gate loads the bank once at startup and exposes a verify() method
  that returns three states: soybean / uncertain / not_soybean.
"""

import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from config import Config
from production_inference_enhanced import SoybeanDiseaseClassifierEnhanced

logger = logging.getLogger(__name__)


class EfficientNetFeatureExtractor(nn.Module):
    """Wrap Enhanced EfficientNet-B3 to expose penultimate embeddings.

    Given the classifier architecture used in production_inference_enhanced
    (efficientnet_b3 with classifier[1] = Linear(in_features, num_classes)),
    we can take the output of classifier[0] (the pre-classifier head) as the
    embedding for similarity-based crop verification.
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # EfficientNet-B3 forward pass (adapted from torchvision implementation)
        # We call the internal blocks manually so we can intercept the feature
        # vector right before the final classification Linear layer.
        x = self.base.features(x)
        x = self.base.avgpool(x)
        x = torch.flatten(x, 1)
        # classifier is typically [Dropout, Linear]. We take the output of the
        # first element (pre-classifier Linear input) as our embedding.
        pre_head = self.base.classifier[0](x)
        return pre_head


class SoybeanEmbeddingVerifier:
    """Embedding-based soybean verification gate.

    Uses cosine similarity between an input leaf embedding and a bank of
    reference soybean embeddings extracted offline from training images.
    """

    def __init__(
        self,
        classifier: SoybeanDiseaseClassifierEnhanced,
        embedding_bank_path: Path,
        high_threshold: float = None,
        low_threshold: float = None,
    ) -> None:
        self.device = classifier.device
        self.classifier = classifier
        self.embedding_bank_path = Path(embedding_bank_path)

        # Thresholds can be tuned via config / environment
        self.high_threshold = (
            high_threshold
            if high_threshold is not None
            else float(getattr(Config, "SOYBEAN_SIM_HIGH", 0.80))
        )
        self.low_threshold = (
            low_threshold
            if low_threshold is not None
            else float(getattr(Config, "SOYBEAN_SIM_LOW", 0.65))
        )

        # Build feature extractor view on top of the primary EfficientNet model
        self.feature_extractor = EfficientNetFeatureExtractor(self.classifier.primary_model).to(
            self.device
        )
        self.feature_extractor.eval()

        # Reuse the same pre-processing pipeline as production classifier
        self.transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Load embedding bank once
        self.embeddings = None  # type: ignore
        self._load_embedding_bank()

    def _load_embedding_bank(self) -> None:
        if not self.embedding_bank_path.exists():
            logger.warning(
                "Soybean embedding bank not found at %s. Verification gate will be permissive.",
                self.embedding_bank_path,
            )
            self.embeddings = None
            return

        data = np.load(self.embedding_bank_path)
        # Expect shape (N, D)
        if data.ndim != 2:
            raise ValueError(
                f"Invalid embedding bank shape {data.shape}, expected 2D (N, D)."
            )

        # L2-normalize reference embeddings
        norms = np.linalg.norm(data, axis=1, keepdims=True) + 1e-10
        self.embeddings = (data / norms).astype(np.float32)
        logger.info(
            "Loaded soybean embedding bank: %d vectors of dim %d from %s",
            self.embeddings.shape[0],
            self.embeddings.shape[1],
            self.embedding_bank_path,
        )

    def _extract_embedding(self, image: Image.Image) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.feature_extractor(tensor)
            # L2 normalize
            feat = torch.nn.functional.normalize(feat, dim=1)
        return feat.squeeze(0)  # (D,)

    def verify(self, image: Image.Image, image_resolution: Tuple[int, int]) -> Dict[str, Any]:
        """Verify whether the input image looks like a soybean leaf.

        Returns a dict:
            {
                "decision": "soybean" | "uncertain" | "not_soybean",
                "max_similarity": float,
                "high_threshold": float,
                "low_threshold": float,
                "image_resolution": (H, W),
                "warning": Optional[str],
            }
        """
        if self.embeddings is None or self.embeddings.size == 0:
            # No embedding bank: log and be permissive but transparent
            logger.warning(
                "Soybean embedding bank is empty or missing. Skipping verification gate.",
            )
            return {
                "decision": "uncertain",
                "max_similarity": None,
                "high_threshold": self.high_threshold,
                "low_threshold": self.low_threshold,
                "image_resolution": image_resolution,
                "warning": "Soybean verification bank not available; skipping identity check.",
            }

        # Extract embedding for current image
        emb = self._extract_embedding(image)  # (D,)
        emb_np = emb.detach().cpu().numpy().astype(np.float32)

        # Cosine similarity via vectorized dot product
        sims = np.dot(self.embeddings, emb_np)  # (N,)
        max_sim = float(sims.max().item())

        # Decision based on thresholds
        if max_sim >= self.high_threshold:
            decision = "soybean"
            warning = None
        elif max_sim >= self.low_threshold:
            decision = "uncertain"
            warning = "Soybean detected with low similarity. Results should be verified."
        else:
            decision = "not_soybean"
            warning = None

        logger.info(
            "Soybean embedding gate: max_similarity=%.4f, decision=%s, image_res=%s, warning=%s",
            max_sim,
            decision,
            image_resolution,
            bool(warning),
        )

        return {
            "decision": decision,
            "max_similarity": max_sim,
            "high_threshold": self.high_threshold,
            "low_threshold": self.low_threshold,
            "image_resolution": image_resolution,
            "warning": warning,
        }
