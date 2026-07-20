"""
Soybean Disease Detection API Backend
Integrates YOLO detection and Enhanced EfficientNet classification models
"""
import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tempfile
import uuid
from ultralytics import YOLO
import base64
from io import BytesIO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the code directory to path to import production modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))

# Import configuration and modules
from config import Config
from utils import allowed_file, validate_and_preprocess_image, crop_and_resize_image, format_detection_result, cleanup_temp_file
from production_inference_enhanced import SoybeanDiseaseClassifierEnhanced
from disease_knowledge import DISEASE_KNOWLEDGE, DATASET_METADATA
from crop_identifier import get_crop_identifier
from llm_reasoning_layer import get_llm_reasoning_layer
from gradcam_visualizer import generate_gradcam_visualization
from soybean_verifier import SoybeanEmbeddingVerifier

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

class SoybeanDiseaseDetectionAPI:
    """Main API class that integrates YOLO detection and Enhanced EfficientNet classification"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize embedding-based soybean verification gate (MANDATORY - HARD FAIL IF MISSING)
        embedding_bank_path = (
            Config.ENHANCED_MODEL_DIR / "soybean_embedding_bank.npy"
        )
        
        # CRITICAL: Reference bank MUST exist - no silent fallback
        if not embedding_bank_path.exists():
            raise FileNotFoundError(
                f"FATAL: Soybean reference embeddings not found at {embedding_bank_path}. "
                "System cannot verify crop identity without reference bank. "
                "Run soybean_embedding_builder.py first."
            )
        
        self.soybean_verifier = SoybeanEmbeddingVerifier(
            classifier=SoybeanDiseaseClassifierEnhanced(model_dir=str(Config.ENHANCED_MODEL_DIR)),
            embedding_bank_path=embedding_bank_path,
            high_threshold=Config.SOYBEAN_SIMILARITY_HIGH,
            low_threshold=Config.SOYBEAN_SIMILARITY_LOW,
            device=self.device
        )
        logger.info(
            "Soybean Embedding Verifier initialized with bank at %s (MANDATORY GATE ACTIVE)",
            embedding_bank_path,
        )
        
        # Initialize Intelligent Crop Identifier (Safety Layer) as fallback only
        self.crop_identifier = None
        
        # Load YOLO model (96.95% mAP model)
        self.yolo_model = YOLO(str(Config.YOLO_MODEL_PATH))
        logger.info("YOLO model loaded successfully")
        
        # Load Enhanced EfficientNet model (98.14% accuracy)
        self.classifier = SoybeanDiseaseClassifierEnhanced(model_dir=str(Config.ENHANCED_MODEL_DIR))
        logger.info("Enhanced EfficientNet model loaded successfully")
        
        # Initialize Dynamic LLM Reasoning Layer (Intelligence Upgrade)
        self.llm_reasoning = get_llm_reasoning_layer(llm_backend='template')
        logger.info("LLM Reasoning Layer initialized (Intelligence Upgrade)")
        
        # Soybean disease classes
        self.disease_classes = [
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
        
        # Disease information for detailed results
        self.disease_info = {
            'Bacterial Blight': {
                'meaning': 'Bacterial blight is a serious disease of soybean caused by the bacterium Pseudomonas syringae pv. glycinea.',
                'symptoms': 'Water-soaked spots on leaves that become angular and brown, with yellow halos. Stems may show dark brown lesions.',
                'recovery': 'Remove and destroy infected plant debris. Use disease-free seeds and resistant varieties when possible.',
                'prevention': 'Rotate crops, avoid overhead irrigation, and ensure good air circulation around plants.',
                'solution': 'Copper-based bactericides may help reduce spread during early infection stages.',
                'treatment': 'Apply copper fungicides at early stages of infection and remove infected plant parts.'
            },
            'Brown Spot': {
                'meaning': 'Brown spot is a fungal disease caused by Septoria glycines that affects soybean leaves.',
                'symptoms': 'Small, dark brown spots with purple borders on lower leaves, leading to defoliation in severe cases.',
                'recovery': 'Improve air circulation and remove infected leaves. Fungicide application may help control spread.',
                'prevention': 'Use resistant varieties, practice crop rotation, and avoid dense plantings.',
                'solution': 'Fungicide applications with active ingredients like strobilurins or triazoles.',
                'treatment': 'Apply fungicides at early reproductive stages (R1-R3) when disease pressure is high.'
            },
            'Caterpillar Pest': {
                'meaning': 'Caterpillars are larvae of moths and butterflies that feed on soybean leaves and pods.',
                'symptoms': 'Holes in leaves, defoliation, and visible caterpillars on plants. May also see frass (caterpillar droppings).',
                'recovery': 'Hand-pick caterpillars when possible. Apply biological controls like Bt (Bacillus thuringiensis).',
                'prevention': 'Monitor fields regularly, encourage natural predators, and use pheromone traps.',
                'solution': 'Biological controls (Bt), spinosad, or synthetic insecticides when thresholds are exceeded.',
                'treatment': 'Apply appropriate insecticides based on pest identification and growth stage.'
            },
            'Ferrugen': {
                'meaning': 'Ferrugen, also known as red leaf, is caused by soybean mosaic virus and environmental stress.',
                'symptoms': 'Reddish-brown discoloration of leaves, often starting at leaf margins and progressing inward.',
                'recovery': 'Remove infected plants and control aphid vectors. Provide adequate nutrition and water.',
                'prevention': 'Use resistant varieties, control aphid populations, and maintain proper plant nutrition.',
                'solution': 'Aphid control with insecticides and cultural practices to reduce stress.',
                'treatment': 'Apply systemic insecticides to control aphid vectors and improve plant health.'
            },
            'Healthy': {
                'meaning': 'The soybean plant shows no signs of disease or pest infestation.',
                'symptoms': 'Normal green coloration, firm structure, and absence of spots, lesions, or deformities.',
                'recovery': 'Maintain current care practices to preserve plant health.',
                'prevention': 'Continue monitoring, proper nutrition, and good agricultural practices.',
                'solution': 'Regular monitoring and maintaining optimal growing conditions.',
                'treatment': 'No treatment needed; continue good management practices.'
            },
            'Mosaic Virus': {
                'meaning': 'Soybean mosaic virus is a viral disease transmitted by aphids and through infected seeds.',
                'symptoms': 'Mosaic patterns of light and dark green on leaves, leaf distortion, and stunted growth.',
                'recovery': 'Remove infected plants to prevent spread. Control aphid populations.',
                'prevention': 'Use virus-free seeds, control aphids, and plant resistant varieties.',
                'solution': 'Integrated pest management focusing on aphid control and resistant varieties.',
                'treatment': 'Remove infected plants and apply insecticides to control aphid vectors.'
            },
            'Rust': {
                'meaning': 'Soybean rust is a fungal disease caused by Phakopsora pachyrhizi.',
                'symptoms': 'Small, tan to reddish-brown pustules on leaf undersides, with yellow halos on upper surfaces.',
                'recovery': 'Apply appropriate fungicides and remove heavily infected leaves.',
                'prevention': 'Monitor fields, use resistant varieties, and apply preventive fungicides.',
                'solution': 'Fungicide applications with triazole or strobilurin active ingredients.',
                'treatment': 'Apply systemic fungicides at early detection and continue monitoring.'
            },
            'Septoria': {
                'meaning': 'Septoria leaf spot is caused by the fungus Septoria glycines.',
                'symptoms': 'Small, dark spots with light centers on leaves, leading to premature defoliation.',
                'recovery': 'Remove infected plant debris and apply fungicides.',
                'prevention': 'Crop rotation, resistant varieties, and proper spacing for air circulation.',
                'solution': 'Fungicide applications when disease pressure is high.',
                'treatment': 'Apply protective fungicides during periods of high humidity and warm temperatures.'
            },
            'Southern Blight': {
                'meaning': 'Southern blight is a fungal disease caused by Sclerotium rolfsii.',
                'symptoms': 'Wilting, yellowing, and death of plants. White fungal growth and small, round sclerotia near the soil line.',
                'recovery': 'Remove infected plants and soil, improve drainage.',
                'prevention': 'Crop rotation, soil solarization, and avoiding excessive moisture.',
                'solution': 'Fungicide soil treatments and cultural practices to reduce humidity.',
                'treatment': 'Apply fungicides like thiophanate-methyl and improve soil drainage.'
            },
            'Sudden Death Syndrome': {
                'meaning': 'Sudden Death Syndrome (SDS) is caused by Fusarium virguliforme.',
                'symptoms': 'Sudden leaf death with interveinal chlorosis and necrosis, brown stem discoloration.',
                'recovery': 'Remove infected plants and rotate to non-host crops.',
                'prevention': 'Use resistant varieties, manage soil moisture, and rotate crops.',
                'solution': 'Resistant varieties and cultural practices to reduce disease pressure.',
                'treatment': 'No effective chemical treatment; focus on cultural controls and resistant varieties.'
            },
            'Vein Necrosis': {
                'meaning': 'Vein necrosis is often associated with virus infections, particularly soybean mosaic virus.',
                'symptoms': 'Brown to black discoloration of leaf veins, often with yellowing of surrounding tissue.',
                'recovery': 'Control virus vectors (aphids) and remove infected plants.',
                'prevention': 'Use resistant varieties and control aphid populations.',
                'solution': 'Integrated pest management for aphid control.',
                'treatment': 'Apply insecticides to control aphid vectors and remove infected plants.'
            },
            'Yellow Mosaic': {
                'meaning': 'Yellow mosaic is a viral disease affecting soybean leaves.',
                'symptoms': 'Yellow mottling and mosaic patterns on leaves, stunted growth, and reduced pod formation.',
                'recovery': 'Remove infected plants and control virus vectors.',
                'prevention': 'Use virus-free seeds and control aphid populations.',
                'solution': 'Aphid control and resistant varieties.',
                'treatment': 'Remove infected plants and apply systemic insecticides to control vectors.'
            }
        }

    def get_enhanced_disease_info(self, disease_name):
        """Format enhanced disease information from knowledge base for API response"""
        knowledge = DISEASE_KNOWLEDGE.get(disease_name, {})
        
        if not knowledge:
            # Fallback to legacy disease_info
            return self.disease_info.get(disease_name, {})
        
        # Extract disease type information
        disease_type_data = knowledge.get('disease_type', {})
        disease_type = disease_type_data.get('classification', 'Unknown')
        causal_organism = disease_type_data.get('causal_organism', '')
        severity = disease_type_data.get('severity', 'Unknown')
        
        # Format symptoms
        symptoms_data = knowledge.get('symptoms', {})
        symptoms_desc = symptoms_data.get('description', '')
        visual_indicators = symptoms_data.get('visual_indicators', [])
        symptoms_text = f"{symptoms_desc}\n\nKey indicators:\n" + "\n".join(f"• {ind}" for ind in visual_indicators)
        
        # Format precautions
        precautions_data = knowledge.get('precautions', {})
        precautions_list = []
        if precautions_data.get('seed_management'):
            precautions_list.append(f"Seed Management: {precautions_data['seed_management']}")
        if precautions_data.get('cultural_practices'):
            precautions_list.append("Cultural Practices:")
            for practice in precautions_data['cultural_practices']:
                precautions_list.append(f"  • {practice}")
        precautions_text = "\n".join(precautions_list)
        
        # Format treatment
        treatment_data = knowledge.get('treatment', {})
        chemical = treatment_data.get('chemical_control', {})
        treatment_list = []
        if chemical.get('products'):
            treatment_list.append("Chemical Control:")
            for product in chemical['products']:
                treatment_list.append(f"  • {product}")
        if chemical.get('application_timing'):
            treatment_list.append(f"\nTiming: {chemical['application_timing']}")
        treatment_text = "\n".join(treatment_list)
        
        # Format fertilizer recommendations
        fertilizer_data = knowledge.get('fertilizer_recommendations', {})
        macro = fertilizer_data.get('macronutrients', {})
        micro = fertilizer_data.get('micronutrients', {})
        organic = fertilizer_data.get('organic_amendments', [])
        
        fertilizer_list = []
        if macro:
            fertilizer_list.append("Macronutrients:")
            for key, value in macro.items():
                if key != 'nitrogen' or isinstance(value, str):
                    fertilizer_list.append(f"  • {key.upper()}: {value}")
        if micro:
            fertilizer_list.append("\nMicronutrients:")
            for key, value in micro.items():
                fertilizer_list.append(f"  • {key.capitalize()}: {value}")
        if organic:
            fertilizer_list.append("\nOrganic Amendments:")
            for amendment in organic:
                fertilizer_list.append(f"  • {amendment}")
        fertilizer_text = "\n".join(fertilizer_list)
        
        return {
            'meaning': f"{disease_type} caused by {causal_organism}" if causal_organism else disease_type,
            'symptoms': symptoms_text,
            'recovery': treatment_data.get('prognosis', 'Integrated management approach recommended'),
            'prevention': precautions_text,
            'solution': treatment_text,
            'treatment': chemical.get('efficacy_note', treatment_text),
            'fertilizers': fertilizer_text,
            'disease_type': disease_type,
            'severity': severity
        }

    def detect_and_classify(self, image_path):
        """Perform detection and classification on an image"""
        try:
            # Validate the image file
            is_valid, message = validate_and_preprocess_image(image_path)
            if not is_valid:
                raise ValueError(message)

            # Load image once for verification and downstream use
            pil_image = Image.open(image_path).convert("RGB")
            img_width, img_height = pil_image.size

            # Similarity-based soybean verification gate
            verifier_result = self.soybean_verifier.verify(
                pil_image,
                image_resolution=(img_height, img_width),
            )

            identity_decision = verifier_result.get("decision", "uncertain")

            # Hard stop if not soybean
            if identity_decision == "not_soybean":
                logger.info(
                    "Soybean verification failed: similarity=%.4f, decision=%s",
                    verifier_result.get("similarity_score", -1.0) or -1.0,
                    identity_decision,
                )
                return {
                    "status": "rejected",
                    "reason": "not_soybean",
                    "similarity_score": verifier_result.get("similarity_score"),
                    "message": "The uploaded leaf does not match soybean leaf patterns from our dataset.",
                    "soybean_verification": verifier_result,
                }

            # Initialize crop identity metadata
            crop_verification = None
            warning_message = verifier_result.get("warning")
            decision = identity_decision
            confidence_state = (
                "high" if identity_decision == "soybean" else "low"
            )
            fusion_score = None
            signals = {
                "similarity_score": verifier_result.get("similarity_score"),
            }

            # Run YOLO detection
            yolo_results = self.yolo_model(
                image_path, 
                imgsz=Config.YOLO_IMGSZ, 
                conf=Config.YOLO_CONFIDENCE, 
                iou=Config.YOLO_IOU_THRESHOLD,
                device=self.device.type if self.device.type == 'cuda' else 'cpu'
            )
            
            # Embedding-based verification is primary; no legacy crop identifier fusion
            logger.info("Using embedding-based verification decision=%s", decision)
            
            detections = []
            for r in yolo_results:
                if r.boxes is not None and len(r.boxes) > 0:
                    xyxy = r.boxes.xyxy.cpu().numpy()  # [N, 4]
                    confs = r.boxes.conf.cpu().numpy()  # [N]
                    classes = r.boxes.cls.cpu().numpy()  # [N]
                    
                    for (x1, y1, x2, y2), score, cls_id in zip(xyxy, confs, classes):
                        detections.append({
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "score": float(score),
                            "class_id": int(cls_id),
                            "class_name": "soybean_disease",  # Single class detector
                        })
            
            # If no detections, return healthy result
            if not detections:
                return {
                    "CLASS": "Healthy",
                    "MEANING": self.disease_info["Healthy"]["meaning"],
                    "SYMPTOMS": self.disease_info["Healthy"]["symptoms"],
                    "RECOVERY": self.disease_info["Healthy"]["recovery"],
                    "PREVENTION": self.disease_info["Healthy"]["prevention"],
                    "SOLUTION": self.disease_info["Healthy"]["solution"],
                    "TREATMENT": self.disease_info["Healthy"]["treatment"],
                    "confidence": 0.95,
                    "model_used": "Enhanced EfficientNet-B3 (Primary)",
                    "crop_type": "soybean",
                    "detections": [],
                    "status": "No disease regions detected by YOLO, classified as healthy"
                }
            
            # 2. For each detection, crop and classify
            image = Image.open(image_path).convert("RGB")
            classification_results = []
            
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                # Ensure valid integer pixel coordinates
                crop_box = (
                    max(0, int(x1)),
                    max(0, int(y1)),
                    max(0, int(x2)),
                    max(0, int(y2)),
                )
                
                # Crop the detected region
                crop = image.crop(crop_box)
                
                # Classify the crop using Enhanced EfficientNet
                classification = self.classifier.predict_from_pil(crop)
                
                classification_results.append({
                    "bbox": det["bbox"],
                    "detection_score": det["score"],
                    "classification": classification
                })
            
            # 3. Return the result with the highest confidence
            best_result = max(classification_results, key=lambda x: x["classification"]["confidence"])
            
            # Get disease info for the best classification
            disease_class = best_result["classification"]["predicted_class"]
            
            # Get enhanced disease information from knowledge base
            enhanced_info = self.get_enhanced_disease_info(disease_class)
            
            # Apply Dynamic LLM Reasoning Layer (Intelligence Upgrade)
            logger.info(f"Applying LLM reasoning for {disease_class}...")
            llm_input = {
                'crop': 'Soybean',
                'disease': disease_class,
                'confidence': best_result["classification"]["confidence"],
                'severity': enhanced_info.get("severity", "Unknown"),
                'static_knowledge': {
                    'meaning': enhanced_info.get("meaning", ""),
                    'symptoms': enhanced_info.get("symptoms", ""),
                    'prevention': enhanced_info.get("prevention", ""),
                    'treatment': enhanced_info.get("treatment", ""),
                    'fertilizers': enhanced_info.get("fertilizers", ""),
                    'precautions': enhanced_info.get("solution", ""),
                    'recovery': enhanced_info.get("recovery", "")
                }
            }
            
            llm_enhanced = self.llm_reasoning.generate_dynamic_advice(llm_input)
            
            # Generate Grad-CAM visualization (Explainability Layer)
            gradcam_result = None
            try:
                logger.info("Generating Grad-CAM heatmap visualization...")
                # Get the cropped image used for classification
                x1, y1, x2, y2 = best_result["bbox"]
                crop_box = (max(0, int(x1)), max(0, int(y1)), max(0, int(x2)), max(0, int(y2)))
                crop = image.crop(crop_box)
                
                # Prepare input tensor for Grad-CAM
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize((512, 512)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                input_tensor = transform(crop.convert('RGB')).unsqueeze(0).to(self.device)
                
                # Get class index for Grad-CAM
                predicted_class_name = best_result["classification"]["predicted_class"]
                try:
                    target_class_idx = self.disease_classes.index(predicted_class_name)
                except ValueError:
                    target_class_idx = None
                
                # Generate Grad-CAM
                gradcam_result = generate_gradcam_visualization(
                    self.classifier.primary_model,
                    crop,
                    input_tensor,
                    target_class=target_class_idx,
                    alpha=0.4
                )
                
                if gradcam_result:
                    logger.info("✓ Grad-CAM visualization generated")
                else:
                    logger.warning("Grad-CAM generation returned None")
                    
            except Exception as e:
                logger.warning(f"Grad-CAM generation failed (non-critical): {e}")
                gradcam_result = None
            
            # Build crop verification metadata (embedding-based primary)
            # Embedding-based soybean verifier is the primary source
            crop_verification_payload = {
                "decision": decision,
                "method": "dataset_similarity_efficientnet",
                "similarity_score": verifier_result.get("similarity_score"),
                "confidence": verifier_result.get("similarity_score"),
                "confidence_state": confidence_state,
                "fusion_score": fusion_score,
                "fusion_max_score": 6,
                "signals": signals,
                "warning": warning_message,
                "status": verifier_result.get("status"),
            }
            
            result = {
                "CLASS": disease_class,
                
                # Dynamic LLM-enhanced fields (NEW)
                "DYNAMIC_EXPLANATION": llm_enhanced.get("dynamic_explanation", enhanced_info.get("meaning", "No information available")),
                "IMMEDIATE_ACTIONS": llm_enhanced.get("immediate_actions", "See treatment section for guidance"),
                "MONITORING_PLAN": llm_enhanced.get("monitoring_plan", "Monitor plant health regularly"),
                "FERTILIZER_GUIDANCE": llm_enhanced.get("fertilizer_guidance", enhanced_info.get("fertilizers", "No recommendations")),
                
                # Original static fields (preserved for backward compatibility)
                "MEANING": enhanced_info.get("meaning", "No information available"),
                "SYMPTOMS": llm_enhanced.get("symptoms", enhanced_info.get("symptoms", "No symptoms described")),
                "RECOVERY": enhanced_info.get("recovery", "No recovery information"),
                "PREVENTION": llm_enhanced.get("precautions", enhanced_info.get("prevention", "No prevention information")),
                "SOLUTION": enhanced_info.get("solution", "No solution provided"),
                "TREATMENT": llm_enhanced.get("treatment", enhanced_info.get("treatment", "No treatment information")),
                "FERTILIZERS": enhanced_info.get("fertilizers", "No fertilizer recommendations"),
                
                "DISEASE_TYPE": enhanced_info.get("disease_type", "Unknown"),
                "SEVERITY": enhanced_info.get("severity", "Unknown"),
                "confidence": best_result["classification"]["confidence"],
                "confidence_percentage": best_result["classification"]["confidence_percentage"],
                "confidence_level": llm_enhanced.get("confidence_level", best_result["classification"]["confidence_level"]),
                "model_used": "Enhanced EfficientNet-B3 (Primary)",
                "crop_type": "soybean",
                "detections": detections,
                "top_predictions": best_result["classification"]["top_predictions"][:3],  # Top 3 predictions
                
                # Metadata
                "llm_reasoning_enabled": True,
                "reasoning_method": llm_enhanced.get("reasoning_method", "template_based"),
                
                # Explainability Layer (Grad-CAM)
                "gradcam_enabled": gradcam_result is not None,
                "gradcam_heatmap": gradcam_result.get("heatmap_overlay") if gradcam_result else None,
                "gradcam_method": gradcam_result.get("visualization_method") if gradcam_result else None,
                
                # Crop Verification Metadata (Safety Gate)
                "crop_verification": crop_verification_payload,
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in detection and classification: {str(e)}")
            raise e

# Initialize the API
detection_api = SoybeanDiseaseDetectionAPI()

# Explicit static file routes - defined BEFORE catch-all route to ensure precedence
import os
frontend_dir_abs = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'code', 'frontend'))
static_dir_abs = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'code', 'frontend', 'static'))

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory(static_dir_abs, filename)

@app.route('/manifest.json')
def serve_manifest():
    """Serve PWA manifest file"""
    return send_from_directory(frontend_dir_abs, 'manifest.json')

@app.route('/service-worker.js')
def serve_service_worker():
    """Serve service worker file"""
    return send_from_directory(frontend_dir_abs, 'service-worker.js')

@app.route('/api/classify', methods=['POST'])
def classify_image():
    """Endpoint for classifying soybean disease from uploaded image"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            # Process the image
            result = detection_api.detect_and_classify(temp_path)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            return jsonify(result)
            
        except Exception as e:
            # Clean up temporary file even if there's an error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            logger.error(f"Classification error: {str(e)}")
            return jsonify({"error": f"Classification failed: {str(e)}"}), 500
            
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route('/api/disease-info', methods=['GET'])
def get_disease_info():
    """Endpoint to get information about all soybean diseases"""
    return jsonify(detection_api.disease_info)

@app.route('/api/disease-info/<disease_name>', methods=['GET'])
def get_specific_disease_info(disease_name):
    """Endpoint to get information about a specific soybean disease"""
    disease_info = detection_api.disease_info.get(disease_name)
    if disease_info:
        return jsonify(disease_info)
    else:
        return jsonify({"error": f"Disease '{disease_name}' not found"}), 404

@app.route('/api/disease-knowledge', methods=['GET'])
def get_disease_knowledge():
    """Endpoint to get enhanced disease knowledge base for LLM/RAG systems"""
    return jsonify({
        'diseases': DISEASE_KNOWLEDGE,
        'metadata': DATASET_METADATA
    })

@app.route('/api/disease-knowledge/<disease_name>', methods=['GET'])
def get_specific_disease_knowledge(disease_name):
    """Endpoint to get detailed knowledge about a specific disease"""
    disease_knowledge = DISEASE_KNOWLEDGE.get(disease_name)
    if disease_knowledge:
        return jsonify({
            'disease': disease_name,
            'data': disease_knowledge
        })
    else:
        return jsonify({"error": f"Disease '{disease_name}' not found in knowledge base"}), 404

@app.route('/')
def serve_index():
    """Serve the main index.html file"""
    return send_from_directory(frontend_dir_abs, 'index.html')

@app.route('/<path:path>')
def serve_frontend(path):
    """Serve frontend files for SPA, excluding API routes"""
    
    # Don't serve API routes through frontend - these should be handled by API endpoints
    if path.startswith('api/'):
        from flask import abort
        abort(404)
    
    # Check if the requested path exists as a file in the frontend directory
    requested_file = os.path.join(frontend_dir_abs, path)
    if os.path.exists(requested_file) and os.path.isfile(requested_file):
        # Serve the specific file
        from pathlib import Path
        directory = str(Path(requested_file).parent)
        filename = Path(requested_file).name
        return send_from_directory(directory, filename)
    else:
        # For any other routes, serve the main index.html (SPA routing)
        return send_from_directory(frontend_dir_abs, 'index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)