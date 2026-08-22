"""
Configuration settings for the Soybean Disease Detection API
"""
import os
from pathlib import Path

class Config:
    """Base configuration class"""
    
    # Project root directory
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    
    # Model paths
    YOLO_MODEL_PATH = PROJECT_ROOT / "experiments" / "runs" / "detect" / "yolo_soybean_disease_training" / "weights" / "best.pt"
    ENHANCED_MODEL_DIR = PROJECT_ROOT / "models" / "CNN_trained_models"
    
    # Detection parameters
    YOLO_IMGSZ = 512
    YOLO_CONFIDENCE = 0.25
    YOLO_IOU_THRESHOLD = 0.45
    
    # Classification parameters
    CLASSIFIER_CONFIDENCE_THRESHOLD = 0.6
    
    # OOD / Reliability parameters (NEW: Prevent Cotton Misclassification)
    RELIABILITY_CONFIDENCE_THRESHOLD = 0.65
    RELIABILITY_ENTROPY_THRESHOLD = 1.5  # Higher than this = Confused/OOD

    # Soybean similarity verification thresholds
    # These control cosine similarity decisions against the soybean embedding bank.
    # High >= SOYBEAN_SIMILARITY_HIGH => soybean confirmed
    # SOYBEAN_SIMILARITY_LOW .. SOYBEAN_SIMILARITY_HIGH => likely soybean (warning)
    # < SOYBEAN_SIMILARITY_LOW => not soybean
    SOYBEAN_SIMILARITY_HIGH = float(os.environ.get("SOYBEAN_SIMILARITY_HIGH", "0.70"))  # ≥ 0.70 → soybean confirmed
    SOYBEAN_SIMILARITY_LOW = float(os.environ.get("SOYBEAN_SIMILARITY_LOW", "0.55"))   # 0.55-0.69 → likely soybean (warning)

    # Disable CLIP gate when using embedding-based verifier
    CLIP_MODEL_NAME = "ViT-B/32"
    CLIP_MARGIN_DELTA = float(os.environ.get("CLIP_MARGIN_DELTA", "0.10"))
    CLIP_GATE_ENABLED = False
    
    # File upload settings
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}
    
    # Disease classes
    DISEASE_CLASSES = [
        'Bacterial Blight',
        'Brown Spot', 
        'Caterpillar Pest',
        'Ferrugen',
        'Healthy',
        'Mosaic Virus',
        'Rust',
        'Septoria',
        'Southern Blight',
        'Sudden Death Syndrome',
        'Vein Necrosis',
        'Yellow Mosaic'
    ]
    
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