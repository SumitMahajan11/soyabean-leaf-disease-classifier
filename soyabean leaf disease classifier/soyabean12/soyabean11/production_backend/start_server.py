"""
Start script for the Soybean Disease Detection API
"""
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Main entry point to start the backend server"""
    print("Starting Soybean Disease Detection API Server...")
    print(f"Project root: {project_root}")
    
    # Check if required models exist
    yolo_model_path = project_root / "runs" / "detect" / "yolo_soybean_disease_training" / "weights" / "best.pt"
    enhanced_model_dir = project_root / "models" / "CNN_trained_models"
    
    if not yolo_model_path.exists():
        print(f"ERROR: YOLO model not found at {yolo_model_path}")
        sys.exit(1)
        
    if not enhanced_model_dir.exists():
        print(f"ERROR: Enhanced EfficientNet model directory not found at {enhanced_model_dir}")
        sys.exit(1)
    
    print("All required models found!")
    print("Loading models and starting server...")
    
    # Import and run the main app
    try:
        from app import app
        print("Models loaded successfully!")
        print("Server starting on http://localhost:5000")
        print("Press Ctrl+C to stop the server")
        
        app.run(host='0.0.0.0', port=5000, debug=False)
        
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()