# Soybean Disease Detection System

A production-ready system for detecting soybean leaf diseases using YOLO detection and Enhanced EfficientNet classification with dataset similarity verification.

## Features

- **Soybean Crop Verification**: Uses 3,256 reference embeddings to verify uploaded images are soybean leaves before processing
- **YOLO Detection**: 96.95% mAP for detecting leaf regions
- **EfficientNet Classification**: 98.14% accuracy for 12 soybean disease classes
- **LLM Reasoning**: Dynamic treatment recommendations
- **Grad-CAM Visualization**: Explainable AI heatmaps

## Architecture

```
Input Image → Soybean Similarity Check → YOLO Detection → EfficientNet Classification → LLM Reasoning → Results
```

## Thresholds

- ≥ 0.70: Soybean confirmed → proceed
- 0.55-0.69: Low confidence → warning
- < 0.55: Not soybean → blocked

## Quick Start

1. Start the server:
```bash
python start_server.py
```

2. Access the web interface at `http://127.0.0.1:5000`

3. Or use the API directly:
```bash
curl -X POST -F "file=@your_image.jpg" http://127.0.0.1:5000/api/classify
```

## Production Inference

```bash
python run_inference.py /path/to/image.jpg
```

## Components

- **Backend**: Flask API with crop verification gate
- **Frontend**: Interactive web interface with tabbed results
- **Models**: YOLOv8n, Enhanced EfficientNet-B3, LLM reasoning layer
- **Verification**: 3,256 soybean reference embeddings