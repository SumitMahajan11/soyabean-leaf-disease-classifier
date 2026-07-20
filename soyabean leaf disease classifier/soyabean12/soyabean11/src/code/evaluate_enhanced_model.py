"""
Quick evaluation script to test the enhanced model
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json

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

def evaluate_model_accuracy(model_name="EnhancedEfficientNet"):
    # Load the enhanced model
    if model_name == "EnhancedEfficientNet":
        model = models.efficientnet_b3(pretrained=False)
        checkpoint_path = "CNN_trained_models/EnhancedEfficientNet/best_model_checkpoint.pth"
    elif model_name == "RefinedEfficientNet":
        model = models.efficientnet_b1(pretrained=False)
        checkpoint_path = "CNN_trained_models/RefinedEfficientNet/best_model_checkpoint.pth"
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 12)  # 12 classes
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create test transform (no augmentation)
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load test dataset
    dataset_path = "data/final_dataset_enhanced"
    if not Path(dataset_path).exists():
        dataset_path = "data/final_dataset"
    
    dataset = SoybeanDiseaseDataset(dataset_path, transform=transform)
    
    # Split dataset into train, validation, test (70%, 15%, 15%)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create test loader
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    # Evaluate
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"{model_name} Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy

if __name__ == "__main__":
    print("Evaluating Enhanced EfficientNet...")
    enhanced_acc = evaluate_model_accuracy("EnhancedEfficientNet")
    
    print("\nEvaluating Refined EfficientNet...")
    refined_acc = evaluate_model_accuracy("RefinedEfficientNet")
    
    print(f"\nSUMMARY:")
    print(f"Original EfficientNet: 94.43%")
    print(f"Enhanced EfficientNet: {enhanced_acc:.4f} ({enhanced_acc*100:.2f}%)")
    print(f"Refined EfficientNet:  {refined_acc:.4f} ({refined_acc*100:.2f}%)")
    
    if enhanced_acc > 0.96:
        print(f"\n🎉 SUCCESS: Enhanced model exceeded 96% target ({enhanced_acc*100:.2f}%)")
    else:
        print(f"\n⚠️  Enhanced model did not reach 96% target")