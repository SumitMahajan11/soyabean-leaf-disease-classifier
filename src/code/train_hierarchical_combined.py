"""
COMBINED TRAINING PIPELINE: EfficientNet-B4 & ResNet152
Strategy: Dampened Sampler + TrivialAugment + Progressive Resizing + Differential LR
Target: >98% accuracy on Soybean Hierarchical Dataset
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import copy
import os

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'='*60}")
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"{'='*60}\n")

class HierarchicalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        tmp_samples = []
        class_set = set()
        for img_path in self.root_dir.rglob("*"):
            if img_path.is_file() and img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                class_name = img_path.parent.name
                tmp_samples.append((img_path, class_name))
                class_set.add(class_name)
        self.classes = sorted(class_set)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = [(img, self.class_to_idx[cls]) for img, cls in tmp_samples]
        print(f"✓ Dataset loaded: {len(self.samples)} images, {len(self.classes)} classes")

    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new('RGB', (512, 512), (0, 0, 0))
        if self.transform: image = self.transform(image)
        return image, label

def get_dampened_sampler(dataset_subset):
    indices = dataset_subset.indices
    labels = [dataset_subset.dataset.samples[i][1] for i in indices]
    class_counts = np.bincount(labels)
    class_weights = 1. / np.sqrt(class_counts + 1e-6)
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        img_path, label = self.subset.dataset.samples[self.subset.indices[index]]
        image = Image.open(img_path).convert("RGB")
        if self.transform: image = self.transform(image)
        return image, label
    def __len__(self): return len(self.subset)

def create_base_model(model_name, num_classes):
    if model_name == "EfficientNet_B4":
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "ResNet152":
        model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)

def run_training_cycle(model_name):
    print(f"\n{'#'*60}")
    print(f"STARTING TRAINING FOR: {model_name}")
    print(f"{'#'*60}")
    
    save_dir = Path(f"e:/soyabean12/soyabean11/models/CNN_trained_models/{model_name}_Final")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Phase 1: 384x384 (Fast Convergence)
    print(f"\n--- [PHASE 1] Initial Training (384x384) ---")
    model = train_step(model_name, img_size=384, num_epochs=20, save_dir=save_dir, stage="P1")
    
    # Phase 2: 512x512 (High Resolution Refinement)
    print(f"\n--- [PHASE 2] Fine-Tuning (512x512) ---")
    train_step(model_name, img_size=512, num_epochs=60, save_dir=save_dir, stage="P2", model=model)

def train_step(model_name, img_size, num_epochs, save_dir, stage, model=None):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.TrivialAugmentWide(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = HierarchicalDataset("e:/soyabean12/soyabean11/final_dataset_enhanced_hierarchical")
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        full_dataset, [int(0.7*len(full_dataset)), int(0.15*len(full_dataset)), len(full_dataset)-int(0.7*len(full_dataset))-int(0.15*len(full_dataset))],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(SubsetWithTransform(train_ds, train_transform), batch_size=8, sampler=get_dampened_sampler(train_ds), num_workers=0)
    val_loader = DataLoader(SubsetWithTransform(val_ds, val_transform), batch_size=8, shuffle=False)

    if model is None:
        model = create_base_model(model_name, len(full_dataset.classes))
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler('cuda')
    
    # Backbone vs Head Learning Rates
    if model_name == "EfficientNet_B4":
        backbone_params = model.features.parameters()
        head_params = model.classifier.parameters()
    else: # ResNet152
        backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
        head_params = model.fc.parameters()

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 2e-5 if stage == "P2" else 5e-5},
        {'params': head_params, 'lr': 1e-4 if stage == "P2" else 2e-4}
    ], weight_decay=1e-3)
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            train_loss += loss.item()
            if (batch_idx + 1) % 100 == 0:
                print(f"    [{stage}] Ep {epoch+1} Batch {batch_idx+1}/{len(train_loader)} Loss: {loss.item():.4f}")
        
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        scheduler.step()
        print(f"  [{stage}] Epoch {epoch+1}/{num_epochs}: Train Acc: {train_correct/train_total:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            print(f"  ★ New Best Acc: {best_acc:.4f} - Saved!")
            
    return model

def main():
    # Train both models sequentially
    run_training_cycle("EfficientNet_B4")
    
    # Clear memory before next model
    torch.cuda.empty_cache()
    
    run_training_cycle("ResNet152")
    
    print("\n[FINISH] Combined Training Complete!")

if __name__ == "__main__":
    main()
