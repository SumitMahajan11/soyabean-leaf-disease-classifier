"""
ULTIMATE TRAINING PIPELINE: EfficientNet-B4
Strategy: WeightedRandomSampler + Two-Stage Fine-Tuning + Label Smoothing + Differential LR
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
from datetime import datetime
import copy
import numpy as np
import os

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'='*60}")
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
print(f"{'='*60}\n")

class HierarchicalDataset(Dataset):
    """Dataset for hierarchical folder structure with recursive scanning"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Collect samples recursively
        tmp_samples = []
        class_set = set()
        
        for img_path in self.root_dir.rglob("*"):
            if img_path.is_file() and img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]:
                class_name = img_path.parent.name  # leaf folder name
                tmp_samples.append((img_path, class_name))
                class_set.add(class_name)
        
        # Build class mapping
        self.classes = sorted(class_set)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = [(img, self.class_to_idx[cls]) for img, cls in tmp_samples]
        
        print(f"✓ Dataset loaded: {len(self.samples)} images, {len(self.classes)} classes")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image or another sample
            image = Image.new('RGB', (512, 512), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
        return image, label

def get_sampler(dataset_subset):
    """Creates a WeightedRandomSampler to solve class imbalance within a subset"""
    indices = dataset_subset.indices
    labels = [dataset_subset.dataset.samples[i][1] for i in indices]
    
    class_counts = np.bincount(labels)
    # Avoid division by zero
    class_weights = 1. / (class_counts + 1e-6)
    sample_weights = [class_weights[l] for l in labels]
    
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

def run_epoch(model, loader, optimizer, criterion, scaler, device, epoch, stage_name, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()
        
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        if is_train:
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
        
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)
        
        if is_train and (batch_idx + 1) % 50 == 0:
            print(f"    Batch {batch_idx+1}/{len(loader)}: Loss: {loss.item():.4f}")
        
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    
    print(f"  {stage_name} - Epoch {epoch+1:2d}: Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
    return epoch_loss, epoch_acc.item()

def main():
    # 1. SETUP DATA
    dataset_path = "e:/soyabean12/soyabean11/final_dataset_enhanced_hierarchical"
    save_dir = Path("e:/soyabean12/soyabean11/models/CNN_trained_models/EfficientNet_B4_Ultimate")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    full_dataset = HierarchicalDataset(dataset_path)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Custom transform wrapper for subsets
    class SubsetWithTransform(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __getitem__(self, index):
            x, y = self.subset[index]
            # Override dataset transform
            img_path, _ = self.subset.dataset.samples[self.subset.indices[index]]
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, y
        def __len__(self):
            return len(self.subset)

    train_loader = DataLoader(SubsetWithTransform(train_ds, train_transform), 
                              batch_size=8, sampler=get_sampler(train_ds), num_workers=0)
    val_loader = DataLoader(SubsetWithTransform(val_ds, val_transform), batch_size=8, shuffle=False)
    test_loader = DataLoader(SubsetWithTransform(test_ds, val_transform), batch_size=8, shuffle=False)
    
    # 2. INITIALIZE MODEL
    print("\n[STEP 1] Initializing EfficientNet-B4...")
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(full_dataset.classes))
    model = model.to(device)
    
    # Label Smoothing Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')
    
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    # 3. STAGE 1: Train Head Only (10 Epochs)
    print("\n[STEP 2] STAGE 1: Training Classifier Head (Backbone Frozen)...")
    for param in model.features.parameters():
        param.requires_grad = False
    
    optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    
    for epoch in range(10):
        t_loss, t_acc = run_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch, "STAGE 1", is_train=True)
        v_loss, v_acc = run_epoch(model, val_loader, None, criterion, scaler, device, epoch, "VAL", is_train=False)
        
        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), save_dir / "best_model.pth")

    # 4. STAGE 2: Full Fine-Tuning (70 Epochs)
    print("\n[STEP 3] STAGE 2: Unfreezing All Layers (Fine-Tuning with Differential LR)...")
    for param in model.parameters():
        param.requires_grad = True
        
    # Differential LR: Backbone 1e-5, Classifier 1e-4
    optimizer = optim.AdamW([
        {'params': model.features.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-3)
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
    
    for epoch in range(70):
        t_loss, t_acc = run_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch, "STAGE 2", is_train=True)
        v_loss, v_acc = run_epoch(model, val_loader, None, criterion, scaler, device, epoch, "VAL", is_train=False)
        
        scheduler.step()
        
        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            print(f"  ★ New Best Accuracy: {best_acc:.4f} (Saved)")

    # 5. EVALUATE ON TEST SET
    print("\n[STEP 4] Evaluating Best Model on Test Set...")
    model.load_state_dict(torch.load(save_dir / "best_model.pth"))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = accuracy_score(all_labels, all_preds)
    print(f"\n{'='*60}")
    print(f"FINAL TEST ACCURACY: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"{'='*60}\n")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=full_dataset.classes))
    
    # Save results
    results = {
        "test_accuracy": test_acc,
        "best_val_accuracy": best_acc,
        "history": history,
        "classes": full_dataset.classes
    }
    with open(save_dir / "ultimate_results.json", 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"All artifacts saved to: {save_dir}")

if __name__ == "__main__":
    main()
