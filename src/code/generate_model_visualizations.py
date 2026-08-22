"""
GENERATE VISUALIZATIONS: Training Curves & Confusion Matrices
For EfficientNet_B4 (V2) and ResNet152_V2
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HierarchicalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        tmp_samples = []
        class_set = set()
        for img_path in self.root_dir.rglob("*"):
            if img_path.is_file() and img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]:
                class_name = img_path.parent.name
                tmp_samples.append((img_path, class_name))
                class_set.add(class_name)
        self.classes = sorted(class_set)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = [(img, self.class_to_idx[cls]) for img, cls in tmp_samples]

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new('RGB', (512, 512), (0, 0, 0))
        if self.transform: image = self.transform(image)
        return image, label

def plot_curves(metrics_path, save_path):
    with open(metrics_path, 'r') as f:
        m = json.load(f)
    
    epochs = range(1, len(m['train_losses']) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, m['train_losses'], 'b-', label='Train Loss')
    plt.plot(epochs, m['val_losses'], 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, m['train_accs'], 'b-', label='Train Acc')
    plt.plot(epochs, m['val_accs'], 'r-', label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✓ Saved curves to {save_path}")

def generate_eval_data(model_name, checkpoint_path, test_loader, class_names, save_dir):
    print(f"\nEvaluating {model_name}...")
    
    # Load model
    if "EfficientNet" in model_name:
        model = models.efficientnet_b4()
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(model.classifier[1].in_features, len(class_names))
        )
    else:
        model = models.resnet152()
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, len(class_names))
        )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 1. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png")
    plt.close()
    
    # 2. Classification Report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    with open(save_dir / "test_report.json", 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"✓ Saved Confusion Matrix and Report to {save_dir}")

def main():
    base_dir = Path("e:/soyabean12/soyabean11/models/CNN_trained_models")
    dataset_path = "e:/soyabean12/soyabean11/final_dataset_enhanced_hierarchical"
    
    # Prepare Dataset & Test Loader
    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    full_dataset = HierarchicalDataset(dataset_path)
    total = len(full_dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size
    
    _, _, test_ds = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    test_ds.dataset.transform = val_transform
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)
    
    # 1. EfficientNet_B4
    eff_dir = base_dir / "EfficientNet_B4"
    if eff_dir.exists():
        plot_curves(eff_dir / "metrics.json", eff_dir / "training_curves.png")
        generate_eval_data("EfficientNet_B4", eff_dir / "best_model_checkpoint.pth", 
                           test_loader, full_dataset.classes, eff_dir)
    
    # 2. ResNet152_V2
    res_dir = base_dir / "ResNet152_V2"
    if res_dir.exists():
        plot_curves(res_dir / "metrics.json", res_dir / "training_curves.png")
        generate_eval_data("ResNet152_V2", res_dir / "best_model_checkpoint.pth", 
                           test_loader, full_dataset.classes, res_dir)

if __name__ == "__main__":
    main()
