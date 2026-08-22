"""
ADVANCED TRAINING V2: ResNet152 on Hierarchical Dataset (17 classes)
Target: >95% accuracy with enhanced techniques
Dataset: final_dataset_enhanced_hierarchical (4255 images, 17 classes)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime
import copy
import numpy as np

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'='*60}")
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
print(f"{'='*60}\n")

class HierarchicalDataset(Dataset):
    """Dataset for hierarchical folder structure"""
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
        except:
            # Handle corrupt images
            image = Image.new('RGB', (512, 512), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
        return image, label

def create_transforms(train_phase=True):
    """Enhanced data augmentation with TrivialAugmentWide"""
    if train_phase:
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.TrivialAugmentWide(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def calculate_class_weights(dataset):
    """Calculate weights for imbalanced dataset"""
    class_counts = [0] * len(dataset.classes)
    for _, label in dataset.samples:
        class_counts[label] += 1
    
    total = len(dataset.samples)
    weights = []
    for count in class_counts:
        weight = total / (len(class_counts) * count) if count > 0 else 0
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32).to(device)

def create_model(num_classes):
    """Create ResNet152 model with enhanced head"""
    model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    # Add dropout for regularization
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)
    )
    return model.to(device)

def train_model(model, train_loader, val_loader, num_epochs=80, save_dir="../../models/CNN_trained_models"):
    """Train with advanced techniques"""
    model_name = "ResNet152_V2"
    save_path = Path(save_dir) / model_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Class weights for imbalanced data
    class_weights = calculate_class_weights(train_loader.dataset.dataset)
    print(f"\n✓ Class weights calculated")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.0002, weight_decay=1e-3)
    
    # Use OneCycleLR for faster and more stable convergence
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.001, epochs=num_epochs, 
        steps_per_epoch=len(train_loader), pct_start=0.1,
        anneal_strategy='cos', div_factor=10, final_div_factor=100
    )
    
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    print(f"\nTraining {model_name} for {num_epochs} epochs\n")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total += inputs.size(0)
            
            if (batch_idx + 1) % 100 == 0:
                print(f'  Batch {batch_idx+1}/{len(train_loader)}: Loss={loss.item():.4f}')
        
        epoch_train_loss = running_loss / total
        epoch_train_acc = running_corrects.double() / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc.item())
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total += inputs.size(0)
        
        epoch_val_loss = running_loss / total
        epoch_val_acc = running_corrects.double() / total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc.item())
        
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}')
        
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'best_acc': best_acc,
                'classes': train_loader.dataset.dataset.classes
            }, save_path / "best_model_checkpoint.pth")
            print(f'  ★ Best model saved! Acc: {epoch_val_acc*100:.2f}%')
    
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), save_path / "model.pth")
    
    # Save metrics
    metrics = {
        "model_name": model_name,
        "best_accuracy": best_acc.item(),
        "best_epoch": best_epoch,
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_losses": val_losses,
        "val_accs": val_accs
    }
    with open(save_path / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)
        
    return model, best_acc

def main():
    dataset_path = "e:/soyabean12/soyabean11/final_dataset_enhanced_hierarchical"
    full_dataset = HierarchicalDataset(dataset_path)
    
    total = len(full_dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size
    
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_ds.dataset.transform = create_transforms(train_phase=True)
    val_ds.dataset.transform = create_transforms(train_phase=False)
    test_ds.dataset.transform = create_transforms(train_phase=False)
    
    batch_size = 8
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    model = create_model(len(full_dataset.classes))
    train_model(model, train_loader, val_loader, num_epochs=80)

if __name__ == "__main__":
    main()
