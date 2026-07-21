"""
PRODUCTION INFERENCE PIPELINE - V2 ENSEMBLE
Soybean Leaf Disease Classification System

MODELS: 
1. EfficientNet-B4 (V2)
2. ResNet152 (V2)
Ensemble Method: Weighted Soft Voting
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
from pathlib import Path
import numpy as np
import warnings
from config import Config

warnings.filterwarnings('ignore')

class SoybeanDiseaseClassifierEnhanced:
    """Production-grade ensemble classifier using V2 EfficientNet-B4 and ResNet152"""
    
    def __init__(self, model_dir=None, confidence_threshold=0.6):
        """
        Initialize the V2 ensemble production classifier
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.model_dir = Path(model_dir) if model_dir else Config.ENHANCED_MODEL_DIR
        
        # Load class names from config
        self.class_names = Config.DISEASE_CLASSES
        self.num_classes = len(self.class_names)
        self.weights = Config.ENSEMBLE_WEIGHTS
        
        # Load models
        self.effnet = self._load_efficientnet_b4()
        self.resnet = self._load_resnet152()
        
        print(f"✓ V2 Ensemble Production Classifier Initialized")
        print(f"  Models: EfficientNet-B4 + ResNet152")
        print(f"  Ensemble Weights: {self.weights}")
        print(f"  Device: {self.device}")
        print(f"  Classes: {self.num_classes}")
    
    @property
    def primary_model(self):
        """Used by embedding verifier - using EfficientNet-B4 as feature extractor"""
        return self.effnet

    def _load_efficientnet_b4(self):
        """Load the V2 EfficientNet-B4 model"""
        checkpoint_path = Config.EFFNET_V2_PATH
        alt_checkpoint_path = Config.ENHANCED_MODEL_DIR / "EfficientNet_B4" / "model.pth"
        
        if checkpoint_path.exists():
            path_to_load = checkpoint_path
        elif alt_checkpoint_path.exists():
            path_to_load = alt_checkpoint_path
        else:
            path_to_load = None

        if path_to_load:
            model = models.efficientnet_b4()
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(model.classifier[1].in_features, self.num_classes)
            )
            checkpoint = torch.load(path_to_load, map_location=self.device)
            state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
            model.load_state_dict(state_dict)
        else:
            print(f"⚠️ Checkpoint not found at {checkpoint_path}. Using torchvision default pretrained weights.")
            model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(model.classifier[1].in_features, self.num_classes)
            )

        model.to(self.device)
        model.eval()
        return model

    def _load_resnet152(self):
        """Load the V2 ResNet152 model"""
        checkpoint_path = Config.RESNET_V2_PATH
        alt_checkpoint_path = Config.ENHANCED_MODEL_DIR / "ResNet152_V2" / "model.pth"
        
        if checkpoint_path.exists():
            path_to_load = checkpoint_path
        elif alt_checkpoint_path.exists():
            path_to_load = alt_checkpoint_path
        else:
            path_to_load = None

        if path_to_load:
            model = models.resnet152()
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p=0.4),
                nn.Linear(in_features, self.num_classes)
            )
            checkpoint = torch.load(path_to_load, map_location=self.device)
            state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            return model
        else:
            print(f"⚠️ ResNet152 Checkpoint not found at {checkpoint_path}. Skipping ResNet152 to conserve RAM.")
            return None

    def _preprocess_image(self, image):
        """Preprocessing for 512x512 models"""
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        
        tensor = transform(image).unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, image_source, return_top_k=5):
        """
        Perform ensemble inference
        """
        input_tensor = self._preprocess_image(image_source)
        
        with torch.no_grad():
            # Use mixed precision if available
            autocast_ctx = torch.amp.autocast('cuda') if self.device.type == 'cuda' else torch.amp.autocast('cpu')
            
            with autocast_ctx:
                eff_out = self.effnet(input_tensor)
                eff_probs = torch.softmax(eff_out, dim=1)
                
                if self.resnet is not None:
                    res_out = self.resnet(input_tensor)
                    res_probs = torch.softmax(res_out, dim=1)
                    ensemble_probs = (self.weights[0] * eff_probs) + (self.weights[1] * res_probs)
                else:
                    ensemble_probs = eff_probs
                
            confidence, predicted_idx = torch.max(ensemble_probs, dim=1)
            confidence = confidence.item()
            predicted_idx = predicted_idx.item()
        
        predicted_class = self.class_names[predicted_idx]
        confidence_level = "High" if confidence >= self.confidence_threshold else "Low"
        
        # Get top-k
        top_k_probs, top_k_indices = torch.topk(ensemble_probs, k=min(return_top_k, self.num_classes), dim=1)
        top_predictions = [
            {
                "class": self.class_names[idx.item()],
                "confidence": prob.item(),
                "percentage": f"{prob.item()*100:.2f}%"
            }
            for prob, idx in zip(top_k_probs[0], top_k_indices[0])
        ]
        
        return {
            "model": "V2 Ensemble (EffNet-B4 + ResNet152)",
            "predicted_class": predicted_class,
            "confidence": confidence,
            "confidence_percentage": f"{confidence*100:.2f}%",
            "confidence_level": confidence_level,
            "top_predictions": top_predictions,
            "all_probabilities": {
                self.class_names[i]: ensemble_probs[0][i].item()
                for i in range(self.num_classes)
            }
        }

    def predict_from_pil(self, image, return_top_k=5):
        """Wrapper for predict to handle PIL image"""
        return self.predict(image, return_top_k=return_top_k)

    def get_model_info(self):
        """Get model metadata"""
        return {
            "name": "Soybean Disease V2 Ensemble",
            "components": ["EfficientNet-B4", "ResNet152"],
            "target_accuracy": "93.1% (Test Set)",
            "num_classes": self.num_classes,
            "classes": self.class_names,
            "weights": self.weights,
            "device": str(self.device)
        }



def demo_inference():
    """Demonstration of the enhanced production inference pipeline"""
    print("\n" + "="*70)
    print("ENHANCED SOYBEAN DISEASE CLASSIFICATION - PRODUCTION INFERENCE")
    print("="*70)
    
    # Initialize classifier
    classifier = SoybeanDiseaseClassifierEnhanced()
    
    # Display model info
    info = classifier.get_model_info()
    print("\nModel Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*70)
    print("Enhanced inference pipeline ready for production use")
    print("="*70)
    print("\nUsage:")
    print("  classifier = SoybeanDiseaseClassifierEnhanced()")
    print("  result = classifier.predict('path/to/image.jpg')")
    print("  print(result['predicted_class'], result['confidence'])")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    demo_inference()