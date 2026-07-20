"""
Grad-CAM Heatmap Visualizer - Explainability Layer
Generates activation heatmaps to show what the CNN focuses on during classification
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import logging
import base64
from io import BytesIO

logger = logging.getLogger(__name__)

class GradCAMVisualizer:
    """
    Generates Grad-CAM heatmaps for EfficientNet-B4 to visualize CNN attention.
    Runs AFTER prediction - visualization only, no training or weight updates.
    """
    
    def __init__(self, model, target_layer=None):
        """
        Initialize Grad-CAM visualizer.
        
        Args:
            model: Trained CNN model (EfficientNet-B4)
            target_layer: Layer to extract gradients from (default: last conv layer)
        """
        self.model = model
        self.model.eval()
        
        # Find target layer (last convolutional layer for EfficientNet)
        if target_layer is None:
            self.target_layer = self._find_target_layer()
        else:
            self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
        
        logger.info(f"Grad-CAM visualizer initialized on layer: {self.target_layer}")
    
    def _find_target_layer(self):
        """Find the last convolutional layer in EfficientNet"""
        # For EfficientNet-B4, the last conv layer is typically in 'features'
        try:
            # EfficientNet structure: model.features[-1] is the last block
            return self.model.features[-1]
        except:
            logger.warning("Could not auto-detect target layer, using fallback")
            return self.model.features
    
    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients"""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Register hooks
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap for given input.
        
        Args:
            input_tensor: Preprocessed input tensor (1, 3, H, W)
            target_class: Class index to generate heatmap for (default: predicted class)
        
        Returns:
            numpy array: Heatmap (H, W) normalized to [0, 1]
        """
        try:
            self.model.zero_grad()
            
            # Forward pass
            output = self.model(input_tensor)
            
            # Get target class
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            
            # Backward pass
            output[0, target_class].backward()
            
            # Generate heatmap
            gradients = self.gradients  # (1, C, H, W)
            activations = self.activations  # (1, C, H, W)
            
            # Global average pooling of gradients
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
            
            # Weighted combination of activation maps
            cam = torch.sum(weights * activations, dim=1).squeeze()  # (H, W)
            
            # ReLU and normalize
            cam = F.relu(cam)
            cam = cam.cpu().numpy()
            
            # Normalize to [0, 1]
            if cam.max() > 0:
                cam = cam / cam.max()
            
            return cam
            
        except Exception as e:
            logger.error(f"Error generating Grad-CAM: {e}")
            return None
    
    def overlay_heatmap(self, original_image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap on original image.
        
        Args:
            original_image: PIL Image or numpy array (H, W, 3)
            heatmap: Heatmap array (h, w) in [0, 1]
            alpha: Transparency of heatmap overlay (0-1)
            colormap: OpenCV colormap
        
        Returns:
            PIL Image: Overlayed visualization
        """
        try:
            # Convert PIL to numpy if needed
            if isinstance(original_image, Image.Image):
                img = np.array(original_image.convert('RGB'))
            else:
                img = original_image.copy()
            
            # Resize heatmap to match image size
            heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            
            # Convert heatmap to RGB using colormap
            heatmap_colored = cv2.applyColorMap(
                (heatmap_resized * 255).astype(np.uint8), 
                colormap
            )
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Blend images
            overlayed = (alpha * heatmap_colored + (1 - alpha) * img).astype(np.uint8)
            
            return Image.fromarray(overlayed)
            
        except Exception as e:
            logger.error(f"Error overlaying heatmap: {e}")
            return original_image if isinstance(original_image, Image.Image) else Image.fromarray(original_image)
    
    def generate_visualization(self, original_image, input_tensor, target_class=None, 
                              alpha=0.4, return_base64=True):
        """
        Complete visualization pipeline: generate heatmap and overlay on image.
        
        Args:
            original_image: PIL Image
            input_tensor: Preprocessed tensor for model
            target_class: Target class for visualization
            alpha: Overlay transparency
            return_base64: If True, return base64-encoded image for web display
        
        Returns:
            dict: {
                'heatmap_overlay': PIL Image or base64 string,
                'confidence': float,
                'target_class': int
            }
        """
        try:
            # Generate heatmap
            heatmap = self.generate_heatmap(input_tensor, target_class)
            
            if heatmap is None:
                logger.warning("Heatmap generation failed, returning None")
                return None
            
            # Overlay on original image
            overlayed_img = self.overlay_heatmap(original_image, heatmap, alpha)
            
            # Convert to base64 for web display
            if return_base64:
                buffered = BytesIO()
                overlayed_img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                heatmap_output = f"data:image/png;base64,{img_str}"
            else:
                heatmap_output = overlayed_img
            
            result = {
                'heatmap_overlay': heatmap_output,
                'visualization_method': 'grad_cam',
                'target_layer': str(self.target_layer.__class__.__name__)
            }
            
            logger.info("Grad-CAM visualization generated successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in visualization pipeline: {e}")
            return None


# Singleton instance
_gradcam_instance = None

def get_gradcam_visualizer(model):
    """
    Get or create Grad-CAM visualizer instance.
    
    Args:
        model: CNN model
    
    Returns:
        GradCAMVisualizer instance
    """
    global _gradcam_instance
    
    if _gradcam_instance is None:
        _gradcam_instance = GradCAMVisualizer(model)
    
    return _gradcam_instance


def generate_gradcam_visualization(model, original_image, input_tensor, 
                                   target_class=None, alpha=0.4):
    """
    Convenience function to generate Grad-CAM visualization.
    
    Args:
        model: Trained CNN
        original_image: PIL Image
        input_tensor: Preprocessed tensor
        target_class: Target class (default: predicted)
        alpha: Overlay transparency
    
    Returns:
        Visualization result dict or None
    """
    try:
        visualizer = get_gradcam_visualizer(model)
        return visualizer.generate_visualization(
            original_image, 
            input_tensor, 
            target_class, 
            alpha
        )
    except Exception as e:
        logger.error(f"Grad-CAM visualization failed: {e}")
        return None
