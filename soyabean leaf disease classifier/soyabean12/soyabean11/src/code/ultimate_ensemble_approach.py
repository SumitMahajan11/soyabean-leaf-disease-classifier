"""
ULTIMATE ENSEMBLE APPROACH FOR SOYBEAN DISEASE CLASSIFICATION
Target: Combine best models to achieve maximum possible accuracy
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SoybeanDiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Get all classes (subdirectories)
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Create list of all image paths and their labels
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for img_path in class_dir.iterdir():
                if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                    self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_test_transform():
    """Create standard test transform (no augmentation)"""
    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class UltimateEnsembleClassifier:
    """Ensemble of our best performing models"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = []
        self.model_weights = []
        self.num_classes = 12
        self.class_names = [
            'Bacterial Blight', 'Brown Spot', 'Caterpillar Pest', 'Ferrugen', 
            'Healthy', 'Mosaic Virus', 'Rust', 'Septoria', 
            'Southern Blight', 'Sudden Death Syndrome', 'Vein Necrosis', 'Yellow Mosaic'
        ]
        
        self._load_models()
    
    def _load_models(self):
        """Load our best performing models"""
        print("Loading best performing models for ensemble...")
        
        # 1. Enhanced EfficientNet-B3 (98.14%)
        try:
            model_b3 = models.efficientnet_b3(pretrained=False)
            model_b3.classifier[1] = nn.Linear(model_b3.classifier[1].in_features, self.num_classes)
            checkpoint_path = "CNN_trained_models/EnhancedEfficientNet/best_model_checkpoint.pth"
            if Path(checkpoint_path).exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                model_b3.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
                model_b3 = model_b3.to(self.device)
                model_b3.eval()
                self.models.append(model_b3)
                self.model_weights.append(0.4)  # Higher weight for best performer
                print("✓ Enhanced EfficientNet-B3 loaded")
            else:
                print("⚠ Enhanced EfficientNet-B3 checkpoint not found")
        except Exception as e:
            print(f"⚠ Could not load Enhanced EfficientNet-B3: {e}")
        
        # 2. Refined EfficientNet-B1 (97.73%)
        try:
            model_b1 = models.efficientnet_b1(pretrained=False)
            model_b1.classifier[1] = nn.Linear(model_b1.classifier[1].in_features, self.num_classes)
            checkpoint_path = "CNN_trained_models/RefinedEfficientNet/best_model_checkpoint.pth"
            if Path(checkpoint_path).exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                model_b1.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
                model_b1 = model_b1.to(self.device)
                model_b1.eval()
                self.models.append(model_b1)
                self.model_weights.append(0.3)  # Medium weight
                print("✓ Refined EfficientNet-B1 loaded")
            else:
                print("⚠ Refined EfficientNet-B1 checkpoint not found")
        except Exception as e:
            print(f"⚠ Could not load Refined EfficientNet-B1: {e}")
        
        # 3. Original EfficientNet-B0 (94.43%)
        try:
            model_b0 = models.efficientnet_b0(pretrained=False)
            model_b0.classifier[1] = nn.Linear(model_b0.classifier[1].in_features, self.num_classes)
            checkpoint_path = "CNN_trained_models/EfficientNet/model.pth"
            if Path(checkpoint_path).exists():
                model_b0.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
                model_b0 = model_b0.to(self.device)
                model_b0.eval()
                self.models.append(model_b0)
                self.model_weights.append(0.15)  # Lower weight
                print("✓ Original EfficientNet-B0 loaded")
            else:
                print("⚠ Original EfficientNet-B0 model not found")
        except Exception as e:
            print(f"⚠ Could not load Original EfficientNet-B0: {e}")
        
        # 4. ResNet-50 (85.98%)
        try:
            model_resnet = models.resnet50(pretrained=False)
            model_resnet.fc = nn.Linear(model_resnet.fc.in_features, self.num_classes)
            checkpoint_path = "CNN_trained_models/ResNet/model.pth"
            if Path(checkpoint_path).exists():
                model_resnet.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
                model_resnet = model_resnet.to(self.device)
                model_resnet.eval()
                self.models.append(model_resnet)
                self.model_weights.append(0.15)  # Lower weight
                print("✓ ResNet-50 loaded")
            else:
                print("⚠ ResNet-50 model not found")
        except Exception as e:
            print(f"⚠ Could not load ResNet-50: {e}")
        
        # Normalize weights
        total_weight = sum(self.model_weights)
        self.model_weights = [w / total_weight for w in self.model_weights]
        
        print(f"Ensemble contains {len(self.models)} models with weights: {self.model_weights}")
    
    def predict(self, image_path):
        """Make prediction using ensemble of models"""
        transform = create_test_transform()
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Get predictions from all models
        ensemble_output = 0
        with torch.no_grad():
            for i, model in enumerate(self.models):
                output = model(image_tensor)
                # Apply softmax and weight the prediction
                prob_output = torch.softmax(output, dim=1) * self.model_weights[i]
                ensemble_output += prob_output
        
        # Get final prediction
        confidence, predicted_idx = torch.max(ensemble_output, dim=1)
        predicted_class = self.class_names[predicted_idx.item()]
        confidence_value = confidence.item()
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence_value,
            "confidence_percentage": f"{confidence_value*100:.2f}%",
            "individual_predictions": self._get_individual_predictions(image_tensor)
        }
    
    def _get_individual_predictions(self, image_tensor):
        """Get predictions from individual models"""
        individual_preds = []
        with torch.no_grad():
            for i, model in enumerate(self.models):
                output = model(image_tensor)
                confidence, predicted_idx = torch.max(torch.softmax(output, dim=1), dim=1)
                individual_preds.append({
                    "model_idx": i,
                    "confidence": confidence.item(),
                    "predicted_class": self.class_names[predicted_idx.item()]
                })
        return individual_preds
    
    def evaluate_ensemble(self, test_loader):
        """Evaluate the ensemble on test data"""
        print("Evaluating ensemble on test set...")
        all_ensemble_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Get ensemble prediction
                ensemble_outputs = 0
                for i, model in enumerate(self.models):
                    output = model(inputs)
                    prob_output = torch.softmax(output, dim=1) * self.model_weights[i]
                    ensemble_outputs += prob_output
                
                _, ensemble_preds = torch.max(ensemble_outputs, 1)
                
                all_ensemble_preds.extend(ensemble_preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_ensemble_preds)
        return accuracy

def main():
    print("Starting Ultimate Ensemble Evaluation")
    print(f"Using device: {device}")
    
    # Initialize ensemble classifier
    ensemble_classifier = UltimateEnsembleClassifier()
    
    if len(ensemble_classifier.models) == 0:
        print("No models loaded for ensemble. Exiting.")
        return
    
    # Load test dataset
    dataset_path = "data/final_dataset_enhanced"
    if not Path(dataset_path).exists():
        dataset_path = "data/final_dataset"
        if not Path(dataset_path).exists():
            raise FileNotFoundError("Dataset not found in either 'data/final_dataset_enhanced' or 'data/final_dataset'")
    
    print(f"Loading test dataset from: {dataset_path}")
    
    # Create full dataset
    full_dataset = SoybeanDiseaseDataset(dataset_path, transform=create_test_transform())
    
    # Split dataset into train, validation, test (70%, 15%, 15%)
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create test loader
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    # Evaluate ensemble
    ensemble_accuracy = ensemble_classifier.evaluate_ensemble(test_loader)
    
    print(f"\n" + "="*60)
    print(f"ULTIMATE ENSEMBLE RESULTS")
    print(f"{'='*60}")
    print(f"Number of models in ensemble: {len(ensemble_classifier.models)}")
    print(f"Ensemble Test Accuracy: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")
    print(f"Individual model weights: {ensemble_classifier.model_weights}")
    
    # Compare with best single model (Enhanced EfficientNet: 98.14%)
    print(f"\nComparison:")
    print(f"Best Single Model: 98.14% (Enhanced EfficientNet-B3)")
    print(f"Ultimate Ensemble: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")
    
    if ensemble_accuracy > 0.9814:
        improvement = ensemble_accuracy - 0.9814
        print(f"✅ ENSEMBLE IMPROVEMENT: +{improvement:.4f} ({improvement*100:.2f}%)")
    else:
        print(f"⚠️  Ensemble did not improve over best single model")
    
    print("="*60)
    
    # Sample prediction demonstration
    print(f"\nSample prediction demonstration:")
    if len(test_dataset) > 0:
        # Get a sample from test dataset
        sample_idx = 0
        sample_path = test_dataset.dataset.samples[sample_idx][0]
        result = ensemble_classifier.predict(sample_path)
        print(f"Sample: {sample_path.name}")
        print(f"Prediction: {result['predicted_class']}")
        print(f"Confidence: {result['confidence_percentage']}")
        print(f"Individual model predictions:")
        for i, pred in enumerate(result['individual_predictions']):
            model_names = ["EffNet-B3", "EffNet-B1", "EffNet-B0", "ResNet-50"]
            model_name = model_names[i] if i < len(model_names) else f"Model-{i}"
            print(f"  {model_name}: {pred['predicted_class']} ({pred['confidence']:.3f})")

if __name__ == "__main__":
    main()