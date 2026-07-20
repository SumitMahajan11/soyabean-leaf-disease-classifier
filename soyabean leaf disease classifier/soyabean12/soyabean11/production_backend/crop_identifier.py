"""
Intelligent Crop Identifier - Safety Layer
Verifies if uploaded image is a soybean leaf before disease detection
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import logging

# Optional CLIP import for semantic crop identity gate
try:
    import clip  # type: ignore
except Exception:
    clip = None

logger = logging.getLogger(__name__)

class SoybeanCropIdentifier:
    """
    Lightweight binary classifier to verify if an image contains a soybean leaf.
    Runs BEFORE YOLO and EfficientNet as a safety gate.
    """
    
    def __init__(self, model_path=None, confidence_threshold=0.35, device=None):
        """
        Initialize the crop identifier.
        
        Args:
            model_path: Path to trained MobileNetV2 checkpoint (optional)
            confidence_threshold: Minimum confidence to classify as soybean (default: 0.35 - permissive)
            device: torch device (cuda/cpu)
        """
        # Three-state thresholds for permissive decision-making
        self.high_confidence_threshold = 0.7  # Definitely soybean
        self.low_confidence_threshold = confidence_threshold  # Minimum to proceed (permissive)
        self.block_threshold = 0.20  # Below this = definitely NOT soybean
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load lightweight MobileNetV2 for binary classification
        self.model = self._build_model()
        
        if model_path and torch.cuda.is_available():
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"Loaded crop identifier model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model checkpoint: {e}. Using visual heuristics.")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"Crop Identifier initialized on {self.device}")
    
    def _build_model(self):
        """Build lightweight MobileNetV2 for binary classification"""
        model = models.mobilenet_v2(pretrained=True)
        
        # Replace classifier head for binary classification
        model.classifier[1] = nn.Linear(model.last_channel, 2)  # 2 classes: soybean, not-soybean
        
        return model
    
    def _visual_heuristics(self, image):
        """
        Fallback visual heuristics for soybean leaf detection.
        Checks basic color and shape characteristics.
        
        Args:
            image: PIL Image
            
        Returns:
            confidence score (0-1)
        """
        try:
            # Convert to numpy for analysis
            img_array = np.array(image.convert('RGB'))
            
            # Check 1: Green dominance (soybean leaves are predominantly green)
            hsv = Image.fromarray(img_array).convert('HSV')
            hsv_array = np.array(hsv)
            
            # Green hue range in HSV (30-90 degrees)
            green_mask = (hsv_array[:,:,0] >= 30) & (hsv_array[:,:,0] <= 90)
            green_ratio = np.sum(green_mask) / (img_array.shape[0] * img_array.shape[1])
            
            # Check 2: Saturation (leaves should have moderate saturation)
            saturation = hsv_array[:,:,1]
            avg_saturation = np.mean(saturation) / 255.0
            
            # Check 3: Not too dark or too bright
            brightness = hsv_array[:,:,2]
            avg_brightness = np.mean(brightness) / 255.0
            
            # Heuristic score calculation
            score = 0.0
            
            # Green dominance (40% weight)
            if green_ratio > 0.3:
                score += 0.4 * min(green_ratio / 0.6, 1.0)
            
            # Moderate saturation (30% weight)
            if 0.2 < avg_saturation < 0.8:
                score += 0.3
            
            # Appropriate brightness (30% weight)
            if 0.3 < avg_brightness < 0.8:
                score += 0.3
            
            return min(score, 0.95)  # Cap at 0.95 for heuristics
            
        except Exception as e:
            logger.error(f"Error in visual heuristics: {e}")
            return 0.0
    
    def _multi_signal_fusion(self, crop_confidence, yolo_detections, image):
        """
        Multi-signal fusion for enhanced crop identification accuracy.
        Combines: crop confidence + YOLO detection + green dominance + area check.
        
        Args:
            crop_confidence: Model/heuristic confidence score
            yolo_detections: YOLO detection results
            image: PIL Image
            
        Returns:
            dict: Fusion result with decision, score, and signal breakdown
        """
        fusion_score = 0
        signals = {
            'crop_conf': 0,
            'yolo_leaf': 0,
            'green_ratio': 0,
            'area_check': 0
        }
        
        # SIGNAL 1: Crop Identifier Confidence (Semantic Signal)
        # Treat as relative score, not strict probability
        if crop_confidence >= 0.6:
            signals['crop_conf'] = 2
            fusion_score += 2
        elif crop_confidence >= 0.4:
            signals['crop_conf'] = 1
            fusion_score += 1
        
        # SIGNAL 2: YOLO Leaf Detection (Strong Structural Signal)
        # If YOLO detects leaf-like regions, it's strong evidence
        yolo_leaf_detected = False
        max_area_ratio = 0.0
        
        if yolo_detections and len(yolo_detections) > 0:
            yolo_leaf_detected = True
            # Calculate largest detection area
            img_width, img_height = image.size
            img_area = img_width * img_height
            
            for det in yolo_detections:
                if hasattr(det, 'boxes') and len(det.boxes) > 0:
                    for box in det.boxes:
                        # Get box dimensions
                        if hasattr(box, 'xyxy'):
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            box_area = (x2 - x1) * (y2 - y1)
                            area_ratio = box_area / img_area
                            max_area_ratio = max(max_area_ratio, area_ratio)
        
        if yolo_leaf_detected:
            signals['yolo_leaf'] = 2
            fusion_score += 2
        
        # SIGNAL 3: Green Color Dominance (Visual Heuristic)
        # Supporting signal only
        green_ratio = self._compute_green_ratio(image)
        if green_ratio >= 0.25:  # At least 25% green pixels
            signals['green_ratio'] = 1
            fusion_score += 1
        
        # SIGNAL 4: Leaf Area Sanity Check (Noise Filter)
        # Reject only if detected region is extremely small
        if max_area_ratio >= 0.02:  # At least 2% of image
            signals['area_check'] = 1
            fusion_score += 1
        elif not yolo_leaf_detected:
            # If no YOLO detection, assume reasonable area
            signals['area_check'] = 1
            fusion_score += 1
        
        # DECISION LOGIC
        # Maximum score: 6 (2+2+1+1)
        # PERMISSIVE: Score ≥3 allows proceeding
        if fusion_score >= 4:
            decision = 'proceed'
            confidence_state = 'high'
            is_soybean = True
        elif fusion_score >= 2:  # LOWERED from 3 - more permissive
            decision = 'proceed_uncertain'
            confidence_state = 'medium'
            is_soybean = True
        else:
            # Only block if score < 2 (very weak signals)
            decision = 'block'
            confidence_state = 'low'
            is_soybean = False
        
        return {
            'decision': decision,
            'confidence_state': confidence_state,
            'is_soybean': is_soybean,
            'fusion_score': fusion_score,
            'signals': signals
        }
    
    def _compute_green_ratio(self, image):
        """
        Compute ratio of green pixels in image.
        
        Args:
            image: PIL Image
            
        Returns:
            float: Green pixel ratio (0-1)
        """
        try:
            # Convert to HSV for better green detection
            img_array = np.array(image.convert('RGB'))
            hsv = Image.fromarray(img_array).convert('HSV')
            hsv_array = np.array(hsv)
            
            # Green hue range in HSV (30-90 degrees)
            green_mask = (hsv_array[:,:,0] >= 30) & (hsv_array[:,:,0] <= 90)
            green_ratio = np.sum(green_mask) / (img_array.shape[0] * img_array.shape[1])
            
            return float(green_ratio)
        except Exception as e:
            logger.warning(f"Error computing green ratio: {e}")
            return 0.0
    
    def verify_crop(self, image_input, yolo_detections=None):
        """
        Main verification method with MULTI-SIGNAL FUSION.
        Combines: model confidence + YOLO detections + visual heuristics + area checks.
        
        Args:
            image_input: PIL Image or path to image file
            yolo_detections: Optional YOLO detection results for signal fusion
            
        Returns:
            dict: {
                'decision': str ('proceed', 'proceed_uncertain', 'block'),
                'is_soybean': bool (for backward compatibility),
                'confidence': float,
                'confidence_state': str ('high', 'medium', 'low'),
                'method': str,
                'fusion_score': int,
                'signals': dict
            }
        """
        try:
            # Load image if path provided
            if isinstance(image_input, str):
                image = Image.open(image_input).convert('RGB')
            else:
                image = image_input.convert('RGB')
            
            # SIGNAL 1: Model/Heuristic Confidence (Semantic Signal)
            try:
                crop_confidence, method = self._model_prediction(image)
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}. Using heuristics.")
                crop_confidence = self._visual_heuristics(image)
                method = 'heuristic'
            
            # MULTI-SIGNAL FUSION (if YOLO detections provided)
            if yolo_detections is not None:
                fusion_result = self._multi_signal_fusion(
                    crop_confidence=crop_confidence,
                    yolo_detections=yolo_detections,
                    image=image
                )
                
                decision = fusion_result['decision']
                confidence_state = fusion_result['confidence_state']
                is_soybean = fusion_result['is_soybean']
                fusion_score = fusion_result['fusion_score']
                signals = fusion_result['signals']
                method = f"{method}+fusion"
                
                logger.info(f"Crop verification (FUSION): decision={decision}, score={fusion_score}/6, signals={signals}")
                
                return {
                    'decision': decision,
                    'is_soybean': is_soybean,
                    'confidence': float(crop_confidence),
                    'confidence_state': confidence_state,
                    'method': method,
                    'fusion_score': fusion_score,
                    'signals': signals
                }
            
            # FALLBACK: Single-signal logic (when YOLO not available yet)
            if crop_confidence >= self.high_confidence_threshold:
                decision = 'proceed'
                confidence_state = 'high'
                is_soybean = True
            elif crop_confidence >= self.low_confidence_threshold:
                decision = 'proceed_uncertain'
                confidence_state = 'medium'
                is_soybean = True
            elif crop_confidence >= self.block_threshold:
                decision = 'proceed_uncertain'
                confidence_state = 'low'
                is_soybean = True
            else:
                decision = 'block'
                confidence_state = 'very_low'
                is_soybean = False
            
            result = {
                'decision': decision,
                'is_soybean': is_soybean,
                'confidence': float(crop_confidence),
                'confidence_state': confidence_state,
                'method': method
            }
            
            logger.info(f"Crop verification (SINGLE): decision={decision}, confidence={crop_confidence:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in crop verification: {e}")
            # PERMISSIVE FALLBACK: On error, allow with warning
            return {
                'decision': 'proceed_uncertain',
                'is_soybean': True,
                'confidence': 0.5,
                'confidence_state': 'unknown_error',
                'method': 'error_fallback',
                'error': str(e)
            }
    
    def _model_prediction(self, image):
        """
        Run MobileNetV2 prediction.
        
        Args:
            image: PIL Image
            
        Returns:
            tuple: (confidence, method)
        """
        # Preprocess
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Class 0: not-soybean, Class 1: soybean
            soybean_confidence = probabilities[0, 1].item()
        
        return soybean_confidence, 'model'
    
    def update_threshold(self, new_threshold):
        """Update confidence threshold"""
        if 0.0 <= new_threshold <= 1.0:
            self.confidence_threshold = new_threshold
            logger.info(f"Threshold updated to {new_threshold}")
        else:
            logger.warning(f"Invalid threshold {new_threshold}. Must be between 0 and 1.")


class CLIPSoybeanIdentityGate:
    """
    CLIP-based open-world crop identity gate.
    Determines whether an input image is a soybean leaf or not-soybean using semantic similarity.
    """

    def __init__(self, device=None, margin_delta=0.10, model_name="ViT-B/32"):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.margin_delta = margin_delta
        self.model_name = model_name
        self.available = False
        self.model = None
        self.preprocess = None
        self.text_features = None
        self.soybean_indices = []
        self.nonsoybean_indices = []

        if clip is None:
            logger.warning("CLIP library is not available; CLIP crop identity gate is disabled.")
            return

        try:
            # Load CLIP model and preprocessing once
            self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            self.model.eval()

            # Prepare and cache text embeddings for soybean vs non-soybean prompts
            soybean_prompts = [
                "a photo of a soybean leaf",
                "a soybean plant leaf in a farm",
                "a close-up soybean leaf",
                "a healthy soybean crop leaf",
            ]
            non_soybean_prompts = [
                "a cotton plant leaf",
                "a wheat plant leaf",
                "a sunflower leaf",
                "a crop leaf that is not soybean",
                "a random plant leaf",
            ]
            all_prompts = soybean_prompts + non_soybean_prompts

            with torch.no_grad():
                text_tokens = clip.tokenize(all_prompts).to(self.device)
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            self.text_features = text_features
            self.soybean_indices = list(range(len(soybean_prompts)))
            self.nonsoybean_indices = list(range(len(soybean_prompts), len(all_prompts)))
            self.available = True

            logger.info(
                f"CLIP Soybean Identity Gate initialized (model={self.model_name}, device={self.device})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize CLIP Soybean Identity Gate: {e}")
            self.available = False

    @property
    def is_available(self):
        return self.available

    def _ensure_available(self):
        if not self.available:
            raise RuntimeError("CLIP crop identity gate is not available")

    def classify_image(self, image):
        """
        Run CLIP-based crop identity check.

        Args:
            image: PIL.Image in RGB
        Returns:
            dict with soybean_score, non_soybean_score, margin, decision
        """
        self._ensure_available()

        if image.mode != "RGB":
            image = image.convert("RGB")

        with torch.no_grad():
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarities = (image_features @ self.text_features.T).squeeze(0)

        soybean_scores = similarities[self.soybean_indices]
        nonsoybean_scores = similarities[self.nonsoybean_indices]

        s_soy = float(torch.max(soybean_scores).item())
        s_non = float(torch.max(nonsoybean_scores).item())
        margin = s_soy - s_non

        # Decision policy based on configurable margin delta
        if margin >= self.margin_delta:
            decision = "soybean"
            confidence_state = "high" if margin >= self.margin_delta * 2 else "medium"
        elif margin <= -self.margin_delta:
            decision = "not_soybean"
            confidence_state = "high" if margin <= -self.margin_delta * 2 else "medium"
        else:
            decision = "uncertain"
            confidence_state = "low"

        logger.info(
            "CLIP crop identity gate: soybean_score=%.4f, non_soybean_score=%.4f, margin=%.4f (delta=%.3f) -> %s",
            s_soy,
            s_non,
            margin,
            self.margin_delta,
            decision,
        )

        return {
            "decision": decision,
            "soybean_score": s_soy,
            "non_soybean_score": s_non,
            "margin": margin,
            "delta": float(self.margin_delta),
            "confidence_state": confidence_state,
            "method": f"clip_{self.model_name}",
        }


# Singleton instance for reuse
_crop_identifier_instance = None

def get_crop_identifier(confidence_threshold=0.6):
    """
    Get singleton instance of crop identifier.
    
    Args:
        confidence_threshold: Minimum confidence for soybean classification
        
    Returns:
        SoybeanCropIdentifier instance
    """
    global _crop_identifier_instance
    
    if _crop_identifier_instance is None:
        _crop_identifier_instance = SoybeanCropIdentifier(
            confidence_threshold=confidence_threshold
        )
    
    return _crop_identifier_instance


def verify_crop(image_input, confidence_threshold=0.6):
    """
    Convenience function for quick crop verification.
    
    Args:
        image_input: PIL Image or path to image
        confidence_threshold: Minimum confidence (default: 0.6)
        
    Returns:
        dict: Verification result
    """
    identifier = get_crop_identifier(confidence_threshold)
    return identifier.verify_crop(image_input)


# Singleton instance for CLIP-based crop identity gate
_clip_gate_instance = None


def get_clip_crop_gate(device=None, margin_delta=None, model_name=None):
    """Get singleton instance of the CLIPSoybeanIdentityGate.

    CLIP is the primary semantic authority for crop identity. This gate
    runs before YOLO and CNN disease classification in an open-world
    soybean vs non-soybean setting.
    """
    global _clip_gate_instance

    if _clip_gate_instance is not None:
        return _clip_gate_instance

    # Lazy import of Config to avoid circular imports at module load time
    try:
        from config import Config  # type: ignore
        default_delta = getattr(Config, "CLIP_MARGIN_DELTA", 0.10)
        default_model = getattr(Config, "CLIP_MODEL_NAME", "ViT-B/32")
    except Exception:
        default_delta = 0.10
        default_model = "ViT-B/32"

    resolved_delta = margin_delta if margin_delta is not None else default_delta
    resolved_model = model_name if model_name is not None else default_model

    gate = CLIPSoybeanIdentityGate(
        device=device,
        margin_delta=resolved_delta,
        model_name=resolved_model,
    )

    _clip_gate_instance = gate
    return _clip_gate_instance
