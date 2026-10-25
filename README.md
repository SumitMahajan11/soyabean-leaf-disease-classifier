# Soybean Leaf Disease Classification System

![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)

A multi-stage deep learning pipeline for soybean leaf diagnosis. It verifies that an uploaded image is actually a soybean leaf, localizes the leaf, classifies it into one of 17 disease/health classes, and returns a diagnosis with a treatment and monitoring plan.

**Verified benchmark:** 92.96% accuracy | 92.29% weighted F1, evaluated on 639 test samples (see [`reports/metrics/efficientnet_b4_test_metrics.json`](reports/metrics/efficientnet_b4_test_metrics.json)).

## How it works

```
Input image
   │
   ▼
[1] Crop verification gate     cosine similarity vs. 3,256 soybean reference embeddings
   │   < 0.55          → blocked (not a soybean leaf)
   │   0.55 to < 0.70  → warning (low-confidence soybean image)
   ▼   ≥ 0.70
[2] YOLO leaf detection        bounding-box localization and ROI extraction
   │
   ▼
[3] EfficientNet-B4 classifier 17 disease/health classes
   │
   ▼
[4] Agricultural reasoning engine   5 expert template variants per disease
   │
   ▼
Diagnosis and treatment plan
```

| Stage | Purpose |
|---|---|
| Crop verification gate | Rejects out-of-distribution images (non-soybean leaves, random photos) before classification, reducing false positives |
| YOLO leaf detection | Crops to the leaf so the classifier sees the region that matters |
| EfficientNet-B4 | Fine-grained classification across 17 classes |
| Reasoning engine | Turns a detection into advice: weather risk, infection timeline, yield/quality impact, field management, disease suppression, adapted to detection confidence and severity |

## Features

- **Out-of-distribution filtering:** low-confidence and non-soybean images are flagged instead of being silently misclassified.
- **17-class classification:** disease classes plus a healthy class (full list below).
- **Template-based reasoning engine:** five curated knowledge variants per disease. It has extensible hooks for local (LLaMA/Mistral) or API-based (OpenAI/Anthropic) LLM backends, which are not wired to a live model by default.
- **Flask API and CLI:** web UI, REST endpoint, and a standalone inference script.

## Supported classes

| | | |
|---|---|---|
| Anthracnose | Bacterial Blight | Bacterial Pustule |
| Brown Spot (Septoria) | Charcoal Rot | Downy Mildew |
| Frogeye Leaf Spot | Healthy | Herbicide Injury |
| Iron Deficiency Chlorosis | Mosaic Virus | Phomopsis Seed Decay |
| Phytophthora Root Rot | Powdery Mildew | Rhizoctonia Aerial Blight |
| Rust | Target Spot | |

## Quick start

**1. Install**

```bash
git clone https://github.com/SumitMahajan11/soyabean-leaf-disease-classifier.git
cd soyabean-leaf-disease-classifier
pip install -r requirements.txt
```

**2. Run the server**

```bash
python production_backend/start_server.py
```

Open the web UI at http://127.0.0.1:5000.

**3. Run inference from the command line**

```bash
python run_inference.py path/to/leaf_image.jpg
```

**4. Call the API**

```bash
curl -X POST -F "file=@sample_leaf.jpg" http://127.0.0.1:5000/api/classify
```

## Repository structure

```
.
├── production_backend/        Flask web API and core modules
│   ├── app.py                 server and endpoint routing
│   ├── crop_identifier.py     embedding-based crop verification
│   ├── disease_knowledge.py   17-class disease knowledge base
│   ├── llm_reasoning_layer.py agricultural reasoning engine
│   └── start_server.py        server entrypoint
├── models/CNN_trained_models/
│   └── EfficientNet_B4/       classifier weights and architecture
├── data/yolo_dataset/
│   └── dataset.yaml           17-class ground-truth specification
├── reports/metrics/
│   └── efficientnet_b4_test_metrics.json   verified test benchmark
├── run_inference.py           standalone CLI inference
└── requirements.txt
```

## Limitations

- The reasoning engine is rule/template-based. LLM backends are supported by design but not connected by default, so advice is curated, not generated.
- The benchmark is reported on the project's own test split (639 samples). No evaluation on field-condition images is reported.
- The crop verification gate is a similarity threshold (0.55 / 0.70). It reduces out-of-distribution errors but does not eliminate them.

## License

MIT. See the [LICENSE](LICENSE) file.
