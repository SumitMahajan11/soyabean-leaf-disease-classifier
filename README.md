# Soybean Leaf Disease Classification System

A production-ready deep learning system for detecting and diagnosing soybean leaf diseases using a multi-stage pipeline: embedding-based crop verification, YOLO object detection & leaf localization, EfficientNet-B4 fine-grained classification across 17 disease classes, and a dynamic agricultural reasoning engine.

## Overview & Performance

- **Classification Model**: Fine-tuned EfficientNet-B4
- **Evaluated Classes**: 17 Soybean Disease / Health Classes
- **Verified Benchmark**: **92.96% Accuracy** | **92.29% Weighted F1-Score** (evaluated on 639 test samples; see `reports/metrics/efficientnet_b4_test_metrics.json`)
- **Crop Verification Gate**: 3,256 reference embeddings for out-of-distribution leaf image filtering

## Key Features

- **Crop Verification Gate**: Embedding-based check confirms an uploaded image is actually a soybean leaf before running disease classification, reducing false positives on out-of-distribution images.
- **Reliability Scoring**: Out-of-distribution / low-confidence detection flagged rather than silently misclassified.
- **Dynamic Agricultural Reasoning Engine**: Multi-variant expert template system (5 curated knowledge variants per disease — weather risk, infection timeline, yield/quality impact, field management, disease suppression) generating tailored treatment and monitoring advice based on detection confidence and severity. Built with extensible hooks for local (LLaMA/Mistral) or API-based (OpenAI/Anthropic) LLM backends, not yet wired to a live model by default.
- **Flask API**: Served as a production inference endpoint supporting web and CLI access.

## Architecture

```
Input Image 
   │
   ▼
[ Crop Verification Gate ] (Cosine similarity check against 3,256 soybean embeddings)
   │
   ├── < 0.55 Similarity  ──► Blocked (Not a Soybean Leaf)
   ├── 0.55 - 0.69        ──► Warning (Low Confidence Soybean Image)
   │
   ▼ (≥ 0.70 Similarity)
[ YOLO Leaf Detection ] (Bounding box localization & ROI extraction)
   │
   ▼
[ EfficientNet-B4 Classification ] (17 Disease Classes, 92.96% Accuracy)
   │
   ▼
[ Dynamic Agricultural Reasoning Engine ] (5 Multi-Variant Expert Templates)
   │
   ▼
[ Diagnosis & Treatment Plan Output ]
```

## Supported 17 Disease Classes

1. Anthracnose
2. Bacterial Blight
3. Bacterial Pustule
4. Brown Spot (Septoria)
5. Charcoal Rot
6. Downy Mildew
7. Frogeye Leaf Spot
8. Healthy
9. Herbicide Injury
10. Iron Deficiency Chlorosis
11. Mosaic Virus
12. Phomopsis Seed Decay
13. Phytophthora Root Rot
14. Powdery Mildew
15. Rhizoctonia Aerial Blight
16. Rust
17. Target Spot

## Quick Start

### 1. Installation & Environment

```bash
git clone https://github.com/SumitMahajan11/soyabean-leaf-disease-classifier.git
cd soyabean-leaf-disease-classifier
pip install -r requirements.txt
```

### 2. Run Production Server

```bash
python production_backend/start_server.py
```
Access the interactive web UI at `http://127.0.0.1:5000`.

### 3. Run Inference via CLI

```bash
python run_inference.py path/to/leaf_image.jpg
```

### 4. API Usage

```bash
curl -X POST -F "file=@sample_leaf.jpg" http://127.0.0.1:5000/api/classify
```

## Repository Structure

```
.
├── production_backend/             # Production Flask Web API & Core Modules
│   ├── app.py                      # Production Flask Server & Endpoint Routing
│   ├── crop_identifier.py          # Embedding-based Crop Verification Engine
│   ├── disease_knowledge.py        # 17-Class Disease Knowledge Base
│   ├── llm_reasoning_layer.py      # Dynamic Agricultural Reasoning Engine
│   └── start_server.py             # Production Server Entrypoint
├── models/                         # Trained Model Checkpoints
│   └── CNN_trained_models/
│       └── EfficientNet_B4/        # EfficientNet-B4 Weights & Architecture
├── data/                           # Dataset Configurations & Embeddings
│   └── yolo_dataset/
│       └── dataset.yaml            # 17-Class Ground Truth Specification
├── reports/
│   └── metrics/
│       └── efficientnet_b4_test_metrics.json # Verified Test Benchmarks
├── run_inference.py                # Standalone CLI Inference Utility
├── requirements.txt                # Dependency Requirements
└── README.md                       # Project Documentation
```

## License

MIT License. See project files for details.