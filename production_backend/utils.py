"""
Utility functions for the Soybean Disease Detection API
"""
import os
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from config import Config
import cv2

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def apply_shadow_removal(image):
    """
    Apply conservative shadow removal to a PIL image in memory.
    Optimized to minimize color shift and preserve disease features.
    """
    try:
        # Convert PIL image to OpenCV format (numpy array)
        img_np = np.array(image)
        # Ensure it's uint8
        if img_np.dtype != np.uint8:
            img_np = img_np.astype(np.uint8)
            
        # Convert RGB to LAB - Luminance is decoupled from color here
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Method: Local Luminance Normalization
        # Use a large kernel to estimate local background illumination
        kernel_size = 31
        # Blur the luminance channel to get background illumination estimate
        l_blur = cv2.GaussianBlur(l, (kernel_size, kernel_size), 0)
        
        # Calculate local difference from average
        l_mean = np.mean(l)
        
        # Identify dark areas (potential shadows)
        # Shadow regions are where local luminance is significantly lower than average
        shadow_mask = cv2.threshold(l_blur, l_mean * 0.85, 255, cv2.THRESH_BINARY_INV)[1]
        
        # Apply gentle local normalization only to shadow areas
        # This brightens shadows while leaving bright areas untouched
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Blend: use enhanced version only where mask is active
        # Use a very subtle blend to prevent artifacts
        l_final = l.copy()
        mask_bool = shadow_mask > 0
        l_final[mask_bool] = cv2.addWeighted(l[mask_bool], 0.85, l_enhanced[mask_bool], 0.15, 0).flatten()
        
        # Merge back
        enhanced_lab = cv2.merge([l_final, a, b])
        result_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # Final safety blend with original: 90% original, 10% enhanced
        # This ensures we don't deviate too far from what the model expects
        output_img = cv2.addWeighted(img_np, 0.90, result_rgb, 0.10, 0)
        
        return Image.fromarray(output_img)
        
    except Exception as e:
        print(f"Error applying shadow removal: {str(e)}")
        # Return original image if shadow removal fails
        return image

def validate_and_preprocess_image(file_path):
    """Validate and preprocess an uploaded image"""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return False, "File does not exist"
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > Config.MAX_CONTENT_LENGTH:
            return False, f"File too large. Maximum size is {Config.MAX_CONTENT_LENGTH / (1024*1024):.1f}MB"
        
        # Validate image format
        try:
            with Image.open(file_path) as img:
                # Verify it's a valid image
                img.verify()
        except Exception:
            return False, "Invalid image file"
        
        return True, "File is valid"
        
    except Exception as e:
        return False, f"Error validating file: {str(e)}"

def crop_and_resize_image(image_path, bbox, target_size=(512, 512)):
    """Crop an image based on bounding box and resize to target size"""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            img = img.convert('RGB')
            
            # Crop based on bounding box
            cropped = img.crop(bbox)
            
            # Resize to target size
            resized = cropped.resize(target_size, Image.Resampling.LANCZOS)
            
            return resized
            
    except Exception as e:
        raise Exception(f"Error cropping and resizing image: {str(e)}")

def get_device():
    """Get the appropriate device for model inference"""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def format_detection_result(detections, classifications):
    """Format detection and classification results for API response"""
    if not detections:
        return {
            "CLASS": "Healthy",
            "confidence": 0.95,
            "model_used": "V2 Ensemble (EffNet-B4 + ResNet152)",
            "crop_type": "soybean",
            "detections": [],
            "status": "No disease regions detected by YOLO, classified as healthy"
        }
    
    # Find the classification with highest confidence
    best_idx = max(range(len(classifications)), key=lambda i: classifications[i]["confidence"])
    
    best_detection = detections[best_idx]
    best_classification = classifications[best_idx]
    
    return {
        "CLASS": best_classification["predicted_class"],
        "confidence": best_classification["confidence"],
        "confidence_percentage": best_classification["confidence_percentage"],
        "confidence_level": best_classification["confidence_level"],
        "model_used": "V2 Ensemble (EffNet-B4 + ResNet152)",
        "crop_type": "soybean",
        "detection_bbox": best_detection["bbox"],
        "detection_score": best_detection["score"],
        "top_predictions": best_classification["top_predictions"][:3],
        "detections": detections
    }

def cleanup_temp_file(file_path):
    """Safely delete a temporary file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
    except Exception as e:
        print(f"Error deleting temporary file {file_path}: {str(e)}")
        return False