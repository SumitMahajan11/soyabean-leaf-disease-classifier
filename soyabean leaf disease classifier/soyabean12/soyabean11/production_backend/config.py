"""
Configuration settings for the Soybean Disease Detection API
"""
import os
from pathlib import Path

class Config:
    """Base configuration class"""
    
    # Project root directory
    PROJECT_ROOT = Path(__file__).parent.parent  # Goes from production_backend/config.py to soyabean11/
    
    # Model paths
    YOLO_MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "yolo_soybean_disease_training" / "weights" / "best.pt"
    ENHANCED_MODEL_DIR = PROJECT_ROOT / "models" / "CNN_trained_models"
    
    # Ensemble Models (V2)
    EFFNET_V2_PATH = ENHANCED_MODEL_DIR / "EfficientNet_B4" / "best_model_checkpoint.pth"
    RESNET_V2_PATH = ENHANCED_MODEL_DIR / "ResNet152_V2" / "best_model_checkpoint.pth"
    ENSEMBLE_WEIGHTS = [0.5, 0.5]  # [EffNet, ResNet]
        
    # Detection parameters
    YOLO_IMGSZ = 512
    YOLO_CONFIDENCE = 0.25  # Restored to standard 0.25 for purity
    YOLO_IOU_THRESHOLD = 0.45
    MAX_DETECTIONS = 20  # Increased to capture all potential symptoms per image
        
    # Grad-CAM / explainability (Disabled by default in CPU production to prevent worker timeouts)
    ENABLE_GRADCAM = os.environ.get("ENABLE_GRADCAM", "false").lower() in ("true", "1", "t")
    ENABLE_SHADOW_REMOVAL = False  # Keep disabled as requested
        
    # Classification parameters
    CLASSIFIER_CONFIDENCE_THRESHOLD = 0.60
        
    # OOD / Reliability parameters (NEW: Prevent Cotton Misclassification)
    RELIABILITY_CONFIDENCE_THRESHOLD = 0.65
    RELIABILITY_ENTROPY_THRESHOLD = 1.5  
        
    # Soybean similarity verification thresholds
    # Reverting to safer thresholds
    SOYBEAN_SIMILARITY_HIGH = float(os.environ.get("SOYBEAN_SIMILARITY_HIGH", "0.70"))  # ≥ 0.70 → soybean confirmed
    SOYBEAN_SIMILARITY_LOW = float(os.environ.get("SOYBEAN_SIMILARITY_LOW", "0.55"))   # 0.55-0.69 → likely soybean (warning)

    # Disable CLIP gate when using embedding-based verifier
    CLIP_MODEL_NAME = "ViT-B/32"
    CLIP_MARGIN_DELTA = float(os.environ.get("CLIP_MARGIN_DELTA", "0.10"))
    CLIP_GATE_ENABLED = False
    
    # File upload settings
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}
    
    # Disease classes (V2 - 17 Classes)
    DISEASE_CLASSES = [
        'Anthracnose',
        'Bacterial Blight',
        'Bacterial Pustule',
        'Brown Spot',
        'Cercospora Leaf Blight',
        'Downey Mildew',
        'Frogeye Leaf Spot',
        'Healthy',
        'Insects',
        'Mosaic Virus',
        'Nutrient Deficiencies',
        'Powdery Mildew',
        'Rust',
        'Southern Blight',
        'Sudden Death Syndrome',
        'Target Spot',
        'Yellow Mosaic'
    ]
    
    # Map folder names to display names (Internal mapping for folder consistency)
    CLASS_MAP = {
        'anthracnose': 'Anthracnose',
        'bacterial_blight': 'Bacterial Blight',
        'bacterial_pustule': 'Bacterial Pustule',
        'brown_spot': 'Brown Spot',
        'cercospora_leaf_blight': 'Cercospora Leaf Blight',
        'downey_mildew': 'Downey Mildew',
        'frogeye_leaf_spot': 'Frogeye Leaf Spot',
        'healthy': 'Healthy',
        'insects': 'Insects',
        'mosaic_virus': 'Mosaic Virus',
        'nutrient_deficiencies': 'Nutrient Deficiencies',
        'powdery_mildew': 'Powdery Mildew',
        'rust': 'Rust',
        'southern_blight': 'Southern Blight',
        'sudden_death_syndrome': 'Sudden Death Syndrome',
        'target_spot': 'Target Spot',
        'yellow_mosaic': 'Yellow Mosaic'
    }
    
    @staticmethod
    def init_app(app):
        """Initialize application with config settings"""
        pass

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}