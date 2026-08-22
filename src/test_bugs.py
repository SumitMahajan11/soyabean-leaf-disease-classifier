"""
Quick bug detection and fix script
Tests all major components
"""
import os
import sys

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

print("=" * 60)
print("SOYBEAN PROJECT BUG DETECTION")
print("=" * 60)

# Test 1: Import all modules
print("\n1. Testing module imports...")
try:
    from backend.config import Config
    print("   ✓ Config imported")
except Exception as e:
    print(f"   ✗ Config import failed: {e}")

try:
    from backend.utils import allowed_file
    print("   ✓ Utils imported")
except Exception as e:
    print(f"   ✗ Utils import failed: {e}")

try:
    from backend.disease_knowledge import DISEASE_KNOWLEDGE
    print("   ✓ Disease knowledge imported")
except Exception as e:
    print(f"   ✗ Disease knowledge import failed: {e}")

try:
    from backend.crop_identifier import get_crop_identifier
    print("   ✓ Crop identifier imported")
except Exception as e:
    print(f"   ✗ Crop identifier import failed: {e}")

try:
    from backend.llm_reasoning_layer import get_llm_reasoning_layer
    print("   ✓ LLM reasoning imported")
except Exception as e:
    print(f"   ✗ LLM reasoning import failed: {e}")

try:
    from backend.gradcam_visualizer import generate_gradcam_visualization
    print("   ✓ Grad-CAM imported")
except Exception as e:
    print(f"   ✗ Grad-CAM import failed: {e}")

# Test 2: Check model files
print("\n2. Checking model files...")
project_root = os.path.dirname(os.path.dirname(__file__))

yolo_path = os.path.join(project_root, "experiments", "runs", "detect", "yolo_soybean_disease_training", "weights", "best.pt")
if os.path.exists(yolo_path):
    print(f"   ✓ YOLO model found: {os.path.getsize(yolo_path) / 1024:.1f} KB")
else:
    print(f"   ✗ YOLO model NOT found at: {yolo_path}")

cnn_path = os.path.join(project_root, "models", "CNN_trained_models", "EnhancedEfficientNet", "best_model_checkpoint.pth")
if os.path.exists(cnn_path):
    print(f"   ✓ EfficientNet model found: {os.path.getsize(cnn_path) / (1024*1024):.1f} MB")
else:
    print(f"   ✗ EfficientNet model NOT found at: {cnn_path}")

# Test 3: Check frontend files
print("\n3. Checking frontend files...")
frontend_files = [
    "src/code/frontend/index.html",
    "src/code/frontend/manifest.json",
    "src/code/frontend/service-worker.js",
    "src/code/frontend/static/i18n-manager.js"
]

for file in frontend_files:
    full_path = os.path.join(project_root, file)
    if os.path.exists(full_path):
        print(f"   ✓ {file}")
    else:
        print(f"   ✗ {file} NOT FOUND")

# Test 4: Check disease knowledge integrity
print("\n4. Checking disease knowledge integrity...")
try:
    from backend.disease_knowledge import DISEASE_KNOWLEDGE
    diseases = list(DISEASE_KNOWLEDGE.keys())
    print(f"   ✓ Knowledge base has {len(diseases)} diseases")
    
    required_fields = ['disease_type', 'symptoms', 'precautions', 'treatment']
    missing = []
    for disease in diseases:
        for field in required_fields:
            if field not in DISEASE_KNOWLEDGE[disease]:
                missing.append(f"{disease}.{field}")
    
    if missing:
        print(f"   ⚠ Missing fields: {missing[:5]}...")
    else:
        print("   ✓ All diseases have required fields")
        
except Exception as e:
    print(f"   ✗ Knowledge check failed: {e}")

# Test 6: Check Reliability Thresholds
print("\n6. Checking Reliability Thresholds...")
try:
    from backend.config import Config
    if hasattr(Config, 'RELIABILITY_CONFIDENCE_THRESHOLD') and hasattr(Config, 'RELIABILITY_ENTROPY_THRESHOLD'):
        print(f"   ✓ RELIABILITY_CONFIDENCE_THRESHOLD: {Config.RELIABILITY_CONFIDENCE_THRESHOLD}")
        print(f"   ✓ RELIABILITY_ENTROPY_THRESHOLD: {Config.RELIABILITY_ENTROPY_THRESHOLD}")
    else:
        print("   ✗ Reliability thresholds NOT found in Config")
except Exception as e:
    print(f"   ✗ Reliability threshold check failed: {e}")

# Test 5: Test disease class mapping
print("\n5. Testing disease class mapping...")
try:
    from backend.config import Config
    from backend.disease_knowledge import DISEASE_KNOWLEDGE
    
    config_classes = set(Config.DISEASE_CLASSES)
    knowledge_classes = set(DISEASE_KNOWLEDGE.keys())
    
    if config_classes == knowledge_classes:
        print(f"   ✓ All {len(config_classes)} classes match")
    else:
        missing_in_knowledge = config_classes - knowledge_classes
        missing_in_config = knowledge_classes - config_classes
        if missing_in_knowledge:
            print(f"   ✗ Classes in config but not in knowledge: {missing_in_knowledge}")
        if missing_in_config:
            print(f"   ✗ Classes in knowledge but not in config: {missing_in_config}")
except Exception as e:
    print(f"   ✗ Class mapping check failed: {e}")

print("\n" + "=" * 60)
print("BUG DETECTION COMPLETE")
print("=" * 60)
