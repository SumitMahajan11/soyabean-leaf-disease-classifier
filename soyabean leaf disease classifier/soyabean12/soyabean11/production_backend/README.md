# Soybean Disease Detection API Backend

## Overview
This is the backend API for the soybean disease detection system that integrates YOLO detection and Enhanced EfficientNet classification models to provide accurate disease identification and analysis.

## Architecture
The backend implements the complete workflow:
```
Input Image → YOLO Detection (High Precision) → Crop Detection → 
Enhanced Preprocessing → V2 Ensemble Classification (~93.11% accuracy) → JSON Output
```

## Features
- **Image Classification API**: Accepts image uploads and returns disease classification results
- **Integrated Pipeline**: Combines YOLO detection with Enhanced EfficientNet classification
- **Detailed Disease Information**: Provides meaning, symptoms, recovery, prevention, and solution information
- **Confidence Scoring**: Returns confidence levels for all predictions
- **Multiple Disease Classes**: Supports 12 different soybean diseases
- **CORS Support**: Ready for frontend integration

## Endpoints

### `GET /`
Health check endpoint that returns API information and model versions.

### `POST /api/classify`
Classifies soybean diseases in an uploaded image.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Field: `file` (image file)

**Response:**
```json
{
  "CLASS": "Rust",
  "MEANING": "Soybean rust is a fungal disease...",
  "SYMPTOMS": "Small, tan to reddish-brown pustules...",
  "RECOVERY": "Apply appropriate fungicides...",
  "PREVENTION": "Monitor fields, use resistant varieties...",
  "SOLUTION": "Fungicide applications with triazole...",
  "TREATMENT": "Apply systemic fungicides...",
  "confidence": 0.92,
  "confidence_percentage": "92.00%",
  "confidence_level": "High",
  "model_used": "Enhanced EfficientNet-B3 (Primary)",
  "crop_type": "soybean",
  "detections": [...],
  "top_predictions": [...]
}
```

### `GET /api/disease-info`
Returns information about all soybean diseases.

### `GET /api/disease-info/<disease_name>`
Returns information about a specific soybean disease.

## Models Used
- **YOLO Detection Model**: High-precision model trained on soybean diseases
- **V2 Ensemble Classification Model**: ~93.11% accuracy for disease classification (EfficientNet-B4 + ResNet152)

## Supported Diseases
1. Bacterial Blight
2. Brown Spot
3. Caterpillar Pest
4. Ferrugen
5. Healthy
6. Mosaic Virus
7. Rust
8. Septoria
9. Southern Blight
10. Sudden Death Syndrome
11. Vein Necrosis
12. Yellow Mosaic

## Configuration
The backend uses a configuration system with different settings for development, production, and testing environments. See `config.py` for all configuration options.

## Requirements
- Python 3.8+
- PyTorch
- Flask
- Ultralytics YOLO
- Pillow
- NumPy

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Ensure model files are in the correct locations:
   - YOLO model: `experiments/runs/detect/yolo_soybean_disease_training/weights/best.pt`
   - EfficientNet model: `models/CNN_trained_models/`
3. Run the server: `python start_server.py`

## File Upload Limits
- Maximum file size: 10MB
- Supported formats: PNG, JPG, JPEG, GIF, BMP, TIFF, WebP

## Error Handling
The API includes comprehensive error handling for:
- Invalid file uploads
- Model loading errors
- Classification errors
- Network issues

## Integration with Frontend
The backend is designed to work seamlessly with the soybean disease detection frontend, providing all necessary endpoints for image classification and disease information retrieval.