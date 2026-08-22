"""
ADVANCED TRAINING V2: EfficientNet-B4 on Hierarchical Dataset (17 classes)
Target: >95% accuracy with enhanced techniques
Dataset: final_dataset_enhanced_hierarchical (4255 images, 17 classes)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
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
        
        # Class distribution
        class_counts = [0] * len(self.classes)
        for _, label in self.samples:
            class_counts[label] += 1
        
        print(f"\nClass distribution:")
        for cls_name, count in zip(self.classes, class_counts):
            print(f"  {cls_name:30s}: {count:4d} ({count/len(self.samples)*100:.1f}%)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
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

def create_model(model_name, num_classes):
    """Create model with enhanced architecture"""
    if model_name == "EfficientNet_B4":
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        # Add dropout for regularization
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
    elif model_name == "ResNet152":
        model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model.to(device)

def train_model(model, model_name, train_loader, val_loader, num_epochs=60, save_dir="../../models/CNN_trained_models"):
    """Train with advanced techniques"""
    save_path = Path(save_dir) / model_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Class weights for imbalanced data
    class_weights = calculate_class_weights(train_loader.dataset.dataset)
    print(f"\n✓ Class weights calculated (min={class_weights.min():.3f}, max={class_weights.max():.3f})")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.0002, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.001, epochs=num_epochs, 
        steps_per_epoch=len(train_loader), pct_start=0.1,
        anneal_strategy='cos', div_factor=10, final_div_factor=100
    )
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    print(f"\n{'='*60}")
    print(f"Training {model_name} for {num_epochs} epochs")
    print(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 40)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Mixed precision training
            if scaler:
                with torch.cuda.amp.autocast():
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
            
            if (batch_idx + 1) % 50 == 0:
                batch_acc = torch.sum(preds == labels.data).item() / inputs.size(0)
                print(f'  Batch {batch_idx+1}/{len(train_loader)}: Loss={loss.item():.4f}, Acc={batch_acc:.4f}')
        
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
                
                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                else:
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
        
        print(f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}')
        print(f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}')
        print(f'LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # Save best model
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'val_acc': epoch_val_acc,
                'classes': train_loader.dataset.dataset.classes
            }, save_path / "best_model_checkpoint.pth")
            print(f'  ★ Best model saved! Acc: {epoch_val_acc:.4f} ({epoch_val_acc*100:.2f}%)')
        
        print(f'Best so far: {best_acc:.4f} at epoch {best_epoch}\n')
    
    print(f'\n{"="*60}')
    print(f'{model_name} Training Complete!')
    print(f'Best Val Acc: {best_acc:.4f} ({best_acc*100:.2f}%) at epoch {best_epoch}')
    print(f'{"="*60}\n')
    
    # Load best weights
    model.load_state_dict(best_model_wts)
    
    # Save final model
    torch.save(model.state_dict(), save_path / "model.pth")
    
    # Save metrics
    metrics = {
        "model_name": model_name,
        "best_accuracy": best_acc.item(),
        "best_epoch": best_epoch,
        "num_epochs": num_epochs,
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_losses": val_losses,
        "val_accs": val_accs
    }
    
    with open(save_path / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)
    
    with open(save_path / "training_log.txt", 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Best Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Completed: {datetime.now()}\n")
    
    return model, best_acc

def evaluate_model(model, test_loader, class_names, model_name, save_dir="../../models/CNN_trained_models"):
    """Evaluate on test set"""
    save_path = Path(save_dir) / model_name
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
            else:
                outputs = model(inputs)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    print(f"\n{'='*60}")
    print(f"{model_name} - Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"{'='*60}\n")
    
    # Save detailed metrics
    detailed = {
        "model_name": model_name,
        "test_accuracy": test_acc,
        "classification_report": report
    }
    
    with open(save_path / "test_metrics.json", 'w') as f:
        json.dump(detailed, f, indent=4)
    
    return test_acc

def main():
    print("ADVANCED TRAINING PIPELINE - Hierarchical Dataset")
    print("Target: >98% accuracy\n")
    
    # Transforms
    train_transform = create_transforms(train_phase=True)
    val_transform = create_transforms(train_phase=False)
    
    # Load dataset
    dataset_path = "e:/soyabean12/soyabean11/final_dataset_enhanced_hierarchical"
    print(f"Loading dataset from: {dataset_path}\n")
    
    full_dataset = HierarchicalDataset(dataset_path)
    
    # Split: 70% train, 15% val, 15% test
    total = len(full_dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size
    
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply transforms
    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = val_transform
    test_ds.dataset.transform = val_transform
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_ds)} ({len(train_ds)/total*100:.1f}%)")
    print(f"  Val:   {len(val_ds)} ({len(val_ds)/total*100:.1f}%)")
    print(f"  Test:  {len(test_ds)} ({len(test_ds)/total*100:.1f}%)")
    
    # DataLoaders (num_workers=0 to avoid Windows multiprocessing issues)
    batch_size = 8 if device.type == 'cuda' else 4
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True if device.type=='cuda' else False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True if device.type=='cuda' else False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True if device.type=='cuda' else False)
    
    class_names = full_dataset.classes
    num_classes = len(class_names)
    
    print(f"\nBatch size: {batch_size}")
    print(f"Number of classes: {num_classes}\n")
    
    # Train EfficientNet-B4
    print(f"\n{'#'*60}")
    print("TRAINING MODEL 1: EfficientNet-B4")
    print(f"{'#'*60}\n")
    
    efficientnet = create_model("EfficientNet_B4", num_classes)
    efficientnet, eff_best_acc = train_model(
        efficientnet, "EfficientNet_B4", 
        train_loader, val_loader, 
        num_epochs=80
    )
    eff_test_acc = evaluate_model(efficientnet, test_loader, class_names, "EfficientNet_B4")
    
    # Final summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE - FINAL RESULTS")
    print(f"{'='*60}")
    print(f"EfficientNet-B4:")
    print(f"  Val Acc:  {eff_best_acc:.4f} ({eff_best_acc*100:.2f}%)")
    print(f"  Test Acc: {eff_test_acc:.4f} ({eff_test_acc*100:.2f}%)")
    print(f"{'='*60}\n")
    
    # Save results
    results = {
        "EfficientNet_B4": {
            "val_accuracy": eff_best_acc.item() if torch.is_tensor(eff_best_acc) else eff_best_acc,
            "test_accuracy": eff_test_acc
        }
    }
    
    save_path = Path("../../models/CNN_trained_models")
    with open(save_path / "training_results.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    print("All models saved to: models/CNN_trained_models/")
    print("Training complete!")

if __name__ == "__main__":
    main()
