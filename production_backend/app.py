"""
Soybean Disease Detection API Backend
Integrates YOLO detection and Enhanced EfficientNet classification models
"""
import os
import sys
import json
import torch
torch.set_num_threads(2)
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

# Add current directory and code directory to path to import production modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
code_dir = os.path.abspath(os.path.join(current_dir, '..', 'code'))
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

# Import configuration and modules
from config import Config
from utils import allowed_file, validate_and_preprocess_image, crop_and_resize_image, format_detection_result, cleanup_temp_file
from production_inference_enhanced import SoybeanDiseaseClassifierEnhanced
from disease_knowledge import DISEASE_KNOWLEDGE, DATASET_METADATA
from soybean_verifier import SoybeanEmbeddingVerifier
from crop_identifier import get_crop_identifier
from llm_reasoning_layer import get_llm_reasoning_layer
from gradcam_visualizer import generate_gradcam_visualization

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

class SoybeanDiseaseDetectionAPI:
    """Main API class that integrates YOLO detection and Enhanced EfficientNet classification"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize embedding-based soybean verification gate
        embedding_bank_path = (
            Config.ENHANCED_MODEL_DIR / "soybean_embedding_bank.npy"
        )
        
        if not embedding_bank_path.exists():
            logger.warning(
                f"Soybean reference embeddings not found at {embedding_bank_path}. "
                "Verification gate will operate in permissive mode."
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
        
        # Load YOLO model (High Precision model)
        try:
            if Config.YOLO_MODEL_PATH.exists() and Config.YOLO_MODEL_PATH.stat().st_size > 0:
                self.yolo_model = YOLO(str(Config.YOLO_MODEL_PATH))
                logger.info("YOLO model loaded successfully")
            else:
                logger.warning("YOLO model not found/empty, skipping detection")
                self.yolo_model = None
        except Exception as e:
            logger.warning(f"YOLO load failed: {e}, continuing without detection")
            self.yolo_model = None
        
        # Load V2 Ensemble Classifier (EffNet-B4 + ResNet152)
        self.classifier = SoybeanDiseaseClassifierEnhanced(model_dir=str(Config.ENHANCED_MODEL_DIR))
        logger.info("V2 Ensemble Classifier loaded successfully")
        
        # Initialize Dynamic LLM Reasoning Layer (Intelligence Upgrade)
        self.llm_reasoning = get_llm_reasoning_layer(llm_backend='template')
        logger.info("LLM Reasoning Layer initialized (Intelligence Upgrade)")
        
        # Soybean disease classes (V2 - 17 Classes)
        self.disease_classes = Config.DISEASE_CLASSES
        self.class_map = Config.CLASS_MAP
        
        # Legacy disease information for detailed results (Internal fallback)
        self.disease_info = {
            'Anthracnose': {
                'meaning': 'Anthracnose is a fungal disease caused by Colletotrichum truncatum, affecting pods, stems, and leaves.',
                'symptoms': 'Irregular brown lesions on stems and pods, premature defoliation.',
                'recovery': 'Destroy crop residue and use certified seeds.',
                'prevention': 'Crop rotation and proper drainage.',
                'solution': 'Apply strobilurin or triazole fungicides.',
                'treatment': 'Fungicide application at reproductive stages.'
            },
            'Bacterial Blight': {
                'meaning': 'Bacterial blight is a serious disease of soybean caused by the bacterium Pseudomonas syringae pv. glycinea.',
                'symptoms': 'Water-soaked spots on leaves that become angular and brown, with yellow halos.',
                'recovery': 'Remove infected debris, use resistant varieties.',
                'prevention': 'Rotate crops and avoid overhead irrigation.',
                'solution': 'Copper-based bactericides.',
                'treatment': 'Apply copper fungicides at early stages.'
            },
            'Bacterial Pustule': {
                'meaning': 'Bacterial Pustule is caused by Xanthomonas axonopodis pv. glycines.',
                'symptoms': 'Small yellow-to-brown spots with raised centers (pustules) on leaf surfaces.',
                'recovery': 'Manage with resistant varieties.',
                'prevention': 'Crop rotation and clean seeds.',
                'solution': 'Usually manageable without chemicals.',
                'treatment': 'Resistant varieties are most effective.'
            },
            'Brown Spot': {
                'meaning': 'Brown spot is a fungal disease caused by Septoria glycines.',
                'symptoms': 'Small, dark brown spots with purple borders on lower leaves.',
                'recovery': 'Improve air circulation, remove infected leaves.',
                'prevention': 'Resistant varieties and crop rotation.',
                'solution': 'Fungicide applications.',
                'treatment': 'Apply fungicides at early reproductive stages.'
            },
            'Cercospora Leaf Blight': {
                'meaning': 'Cercospora Leaf Blight is caused by Cercospora kikuchii.',
                'symptoms': 'Purple-to-bronze discoloration of upper leaves in sun-exposed canopy.',
                'recovery': 'Rotate crops and manage residue.',
                'prevention': 'Use treated seeds and scout early.',
                'solution': 'Apply QoI or DMI fungicides.',
                'treatment': 'Fungicide application at R3-R5 stages.'
            },
            'Downey Mildew': {
                'meaning': 'Downey Mildew is a common fungal disease caused by Peronospora manshurica.',
                'symptoms': 'Pale green to yellow spots on upper leaves, gray fuzzy growth underneath.',
                'recovery': 'Generally low impact on yield.',
                'prevention': 'Seed treatment and crop rotation.',
                'solution': 'Foliar fungicides if severe.',
                'treatment': 'Rarely requires chemical treatment.'
            },
            'Frogeye Leaf Spot': {
                'meaning': 'Frogeye Leaf Spot is caused by Cercospora sojina.',
                'symptoms': 'Circular spots with gray centers and dark reddish-brown borders.',
                'recovery': 'Rotation and resistant varieties.',
                'prevention': 'Use varieties with Rcs3 gene.',
                'solution': 'Apply strobilurin or triazole fungicides.',
                'treatment': 'Fungicide at R3 stage if present.'
            },
            'Healthy': {
                'meaning': 'The soybean plant shows no signs of disease or pest infestation.',
                'symptoms': 'Normal green coloration and vigorous growth.',
                'recovery': 'Maintain current care practices.',
                'prevention': 'Continue routine monitoring.',
                'solution': 'None needed.',
                'treatment': 'No treatment required.'
            },
            'Insects': {
                'meaning': 'Insect damage including caterpillars, beetles, or aphids.',
                'symptoms': 'Holes in leaves, defoliation, or yellowing/curling.',
                'recovery': 'Monitor and apply controls if thresholds met.',
                'prevention': 'Encourage natural predators and scouting.',
                'solution': 'Appropriate insecticides or biological controls.',
                'treatment': 'Spray based on economic thresholds.'
            },
            'Mosaic Virus': {
                'meaning': 'Soybean Mosaic Virus is transmitted by aphids and seeds.',
                'symptoms': 'Mosaic patterns of light/dark green, leaf distortion.',
                'recovery': 'Remove infected plants.',
                'prevention': 'Virus-free seeds and aphid control.',
                'solution': 'No chemical cure; focus on prevention.',
                'treatment': 'Vector control and roguing.'
            },
            'Nutrient Deficiencies': {
                'meaning': 'Lack of essential nutrients like Nitrogen, Phosphorus, Potassium, or Iron.',
                'symptoms': 'Interveinal yellowing, stunted growth, or purple tints.',
                'recovery': 'Apply missing nutrients via fertilizer.',
                'prevention': 'Regular soil testing.',
                'solution': 'Balanced fertilization program.',
                'treatment': 'Soil or foliar nutrient application.'
            },
            'Powdery Mildew': {
                'meaning': 'Powdery Mildew is caused by Microsphaera diffusa.',
                'symptoms': 'White powdery growth on leaves, stems, and pods.',
                'recovery': 'Sulfur-based fungicides.',
                'prevention': 'Resistant varieties.',
                'solution': 'Fungicide if infection is early/severe.',
                'treatment': 'Apply triazoles or sulfur fungicides.'
            },
            'Rust': {
                'meaning': 'Soybean rust is a highly destructive fungal disease.',
                'symptoms': 'Small tan/brown pustules on leaf undersides.',
                'recovery': 'Fast action with fungicides required.',
                'prevention': 'Early planting and monitoring.',
                'solution': 'Triazole + Strobilurin combinations.',
                'treatment': 'Preventive application at R1-R2.'
            },
            'Southern Blight': {
                'meaning': 'Southern Blight is a soil-borne fungal disease causing rapid wilt.',
                'symptoms': 'White fungal growth and brown sclerotia at soil line.',
                'recovery': 'Difficult; remove infected plants.',
                'prevention': 'Deep plowing and crop rotation.',
                'solution': 'Soil fungicides.',
                'treatment': 'Preventive soil drenching.'
            },
            'Sudden Death Syndrome': {
                'meaning': 'SDS is a soil-borne disease caused by Fusarium virguliforme.',
                'symptoms': 'Interveinal chlorosis/necrosis, root rot.',
                'recovery': 'Manage with resistant varieties.',
                'prevention': 'Long rotations and compaction reduction.',
                'solution': 'Seed treatments (Fluopyram).',
                'treatment': 'No effective foliar fungicides.'
            },
            'Target Spot': {
                'meaning': 'Target Spot causes circular lesions with concentric rings.',
                'symptoms': 'Target-like spots on leaves, premature defoliation.',
                'recovery': 'Manage humidity and spacing.',
                'prevention': 'Scouting and rotation.',
                'solution': 'Strobilurin or SDHI fungicides.',
                'treatment': 'Fungicide at R1-R3 if humid.'
            },
            'Yellow Mosaic': {
                'meaning': 'Yellow Mosaic is a viral disease transmitted by whiteflies.',
                'symptoms': 'Bright yellow mosaic patterns and stunting.',
                'recovery': 'Remove infected plants.',
                'prevention': 'Whitefly control and resistant varieties.',
                'solution': 'No cure; focus on whitefly management.',
                'treatment': 'Systemic insecticides for vectors.'
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

            # 1. Run YOLO detection (if available)
            detections = []
            if self.yolo_model is not None:
                yolo_results = self.yolo_model(
                    image_path, 
                    imgsz=Config.YOLO_IMGSZ, 
                    conf=Config.YOLO_CONFIDENCE, 
                    iou=Config.YOLO_IOU_THRESHOLD,
                    device=self.device.type if self.device.type == 'cuda' else 'cpu'
                )
                
                # Embedding-based verification is primary; no legacy crop identifier fusion
                logger.info("Using embedding-based verification decision=%s", decision)
                
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

                # Keep only top-k detections for classification to reduce latency
                max_dets = getattr(Config, "MAX_DETECTIONS", 5)
                if len(detections) > max_dets:
                    detections = sorted(detections, key=lambda d: d["score"], reverse=True)[:max_dets]
                
                logger.info(f"YOLO found {len(detections)} potential disease regions.")
            else:
                logger.warning("YOLO model not loaded; using full image as single detection region.")
                detections.append({
                    "bbox": [0.0, 0.0, float(img_width), float(img_height)],
                    "score": 1.0,
                    "class_id": 0,
                    "class_name": "full_image_fallback",
                })
            
            # 2. Strict Detection Logic: 
            # If YOLO fails to find regions, evaluate the full image as a neutral backup.
            is_fallback = False
            if not detections:
                logger.info("YOLO found no spots. Using Full Image as backup.")
                is_fallback = True
                detections.append({
                    "bbox": [0.0, 0.0, float(img_width), float(img_height)],
                    "score": 1.0,
                    "class_id": 0,
                    "class_name": "full_image_backup",
                })
            
            # 3. For each detection, crop (zero padding for purity) and classify
            image = pil_image
            classification_results = []
            padding_ratio = 0.0 # Strict alignment with YOLO bounding box
            
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                # Strict crop coordinates
                padded_x1 = max(0, int(x1))
                padded_y1 = max(0, int(y1))
                padded_x2 = min(img_width, int(x2))
                padded_y2 = min(img_height, int(y2))

                crop_box = (
                    padded_x1,
                    padded_y1,
                    padded_x2,
                    padded_y2,
                )
                
                # Crop the detected region
                crop = image.crop(crop_box)
                
                # Classify the crop using V2 Ensemble
                classification = self.classifier.predict_from_pil(crop)
                
                classification_results.append({
                    "bbox": det["bbox"],
                    "padded_bbox": [padded_x1, padded_y1, padded_x2, padded_y2],
                    "detection_score": det["score"],
                    "detector_class": det["class_name"],
                    "classification": classification
                })
            
            # 4. Objective Result Selection:
            # Select the classification with the highest absolute confidence.
            # No bias towards disease; pure statistical reporting.
            best_result = max(classification_results, key=lambda x: x["classification"]["confidence"])
            
            if best_result["classification"]["predicted_class"] == "Healthy":
                status_msg = "Confirmed healthy by V2 Ensemble." if not is_fallback else "No regions detected; full image verified healthy."
            else:
                status_msg = f"Detected {best_result['classification']['predicted_class']} in localized spot."
            
            # Get disease info for the best classification
            disease_class = best_result["classification"]["predicted_class"]
            
            logger.info(f"Final Objective Prediction: {disease_class} (Confidence: {best_result['classification']['confidence']:.2f})")

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
            if getattr(Config, "ENABLE_GRADCAM", False):
                try:
                    logger.info("Generating Grad-CAM heatmap visualization...")
                    # Get the cropped image used for classification
                    x1, y1, x2, y2 = best_result["bbox"]
                    crop_box = (max(0, int(x1)), max(0, int(y1)), max(0, int(x2)), max(0, int(y2)))
                    crop = image.crop(crop_box)
                    
                    # Prepare input tensor for Grad-CAM
                    from torchvision import transforms
                    transform = transforms.Compose([
                        transforms.Resize((256, 256)),
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
            
            # Build enriched per-detection results and global summary
            detection_results = []
            disease_counts = {}

            for item in classification_results:
                cls = item["classification"]
                disease_name = cls["predicted_class"]
                detection_results.append({
                    "bbox": item["bbox"],
                    "padded_bbox": item["padded_bbox"],
                    "detector_class": item.get("detector_class", "soybean_disease"),
                    "detector_score": item["detection_score"],
                    "disease_class": disease_name,
                    "disease_confidence": cls["confidence"],
                    "disease_confidence_percentage": cls.get("confidence_percentage"),
                    "disease_confidence_level": cls.get("confidence_level"),
                    "top_predictions": cls.get("top_predictions", [])
                })
                disease_counts[disease_name] = disease_counts.get(disease_name, 0) + 1

            summary = {
                "total_detections": len(detection_results),
                "disease_counts": disease_counts
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
                "model_used": "V2 Ensemble (EffNet-B4 + ResNet152)",
                "crop_type": "soybean",
                "detections": detection_results,
                "summary": summary,
                "status": status_msg,
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
                "soybean_verification": crop_verification_payload,
                
                # LLM reasoning metadata (structured)
                "llm_reasoning": {
                    "dynamic_explanation": llm_enhanced.get("dynamic_explanation", enhanced_info.get("meaning", "No information available")),
                    "immediate_actions": llm_enhanced.get("immediate_actions", "See treatment section for guidance"),
                    "monitoring_plan": llm_enhanced.get("monitoring_plan", "Monitor plant health regularly"),
                    "fertilizer_guidance": llm_enhanced.get("fertilizer_guidance", enhanced_info.get("fertilizers", "No recommendations")),
                    "severity": enhanced_info.get("severity", "Unknown"),
                    "reasoning_method": llm_enhanced.get("reasoning_method", "template_based")
                },
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in detection and classification: {str(e)}")
            raise e

# Initialize the API
detection_api = SoybeanDiseaseDetectionAPI()

# Explicit static file routes - defined BEFORE catch-all route to ensure precedence
import os
frontend_dir_abs = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
static_dir_abs = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))

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