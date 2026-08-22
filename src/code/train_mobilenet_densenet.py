import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np
from PIL import Image
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from datetime import datetime
import copy
import warnings
warnings.filterwarnings('ignore')

# Auto-detect GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

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
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            with Image.open(img_path) as img:
                image = img.convert('RGB').copy()
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (512, 512), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_preprocessing_transforms(is_training=False):
    """Create unified preprocessing pipeline with optional data augmentation"""
    if is_training:
        # Training with data augmentation for better generalization
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation/Test without augmentation
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def create_model(model_name, num_classes):
    """Create and initialize the specified model"""
    if model_name.lower() == 'mobilenet':
        # Use MobileNetV3 Large for better performance
        try:
            model = models.mobilenet_v3_large(pretrained=True)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        except:
            # Fallback to MobileNetV2
            model = models.mobilenet_v2(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name.lower() == 'densenet':
        # Use DenseNet121 for memory efficiency
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model.to(device)

def train_model(model, train_loader, val_loader, num_epochs, model_name, save_dir="CNN_trained_models"):
    """Train a single model with early stopping and LR scheduling"""
    save_path = Path(save_dir) / model_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    criterion = nn.CrossEntropyLoss()
    
    # Model-specific hyperparameters
    if model_name.lower() == 'mobilenet':
        lr = 0.0005  # Slightly higher LR for MobileNet
        patience = 8
    else:  # DenseNet
        lr = 0.0001  # Lower LR for DenseNet stability
        patience = 10
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=patience//2, min_lr=1e-7
    )
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    print(f"\n{'='*70}")
    print(f"Training {model_name} with {num_epochs} epochs (max)")
    print(f"Learning rate: {lr}, Patience: {patience}")
    print(f"{'='*70}\n")
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 70)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
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
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # Clear cache periodically
            if batch_idx % 50 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = running_corrects.double() / len(train_loader.dataset)
        
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc.item())
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
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
        
        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = running_corrects.double() / len(val_loader.dataset)
        
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc.item())
        
        print(f'Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}')
        print(f'Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.4f}')
        
        # Track best model
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            patience_counter = 0
            print(f'✓ New best model saved! Best Val Acc: {best_acc:.4f}')
        else:
            patience_counter += 1
            print(f'No improvement. Patience: {patience_counter}/{patience}')
        
        # Learning rate scheduling
        scheduler.step(epoch_val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Learning rate: {current_lr:.2e}')
        print()
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    print(f'\nTraining completed! Best Val Acc: {best_acc:.4f} at epoch {best_epoch}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Save model
    torch.save(model.state_dict(), save_path / "model.pth")
    
    # Save training metrics
    metrics = {
        "model_name": model_name,
        "best_accuracy": best_acc.item(),
        "best_epoch": best_epoch,
        "total_epochs_run": len(train_losses),
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies
    }
    
    with open(save_path / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save training log
    with open(save_path / "training_log.txt", 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Best Validation Accuracy: {best_acc:.4f}\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Total Epochs Run: {len(train_losses)}\n")
        f.write(f"Training completed at: {datetime.now()}\n")
    
    # Plot training curves
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label='Train Accuracy', marker='o', markersize=3)
    plt.plot(val_accuracies, label='Validation Accuracy', marker='s', markersize=3)
    plt.axvline(x=best_epoch-1, color='r', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch})')
    plt.title(f'{model_name} - Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss', marker='o', markersize=3)
    plt.plot(val_losses, label='Validation Loss', marker='s', markersize=3)
    plt.axvline(x=best_epoch-1, color='r', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch})')
    plt.title(f'{model_name} - Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / "training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return model, best_acc

def evaluate_model(model, test_loader, class_names, model_name, save_dir="CNN_trained_models"):
    """Comprehensive model evaluation"""
    save_path = Path(save_dir) / model_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    print(f"\nEvaluating {model_name} on test set...")
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
            else:
                outputs = model(inputs)
            
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate comprehensive metrics
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Save detailed metrics
    metrics = {
        "model_name": model_name,
        "test_accuracy": accuracy,
        "classification_report": report
    }
    
    with open(save_path / "detailed_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'{model_name} - Confusion Matrix\nTest Accuracy: {accuracy:.4f}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path / "confusion_matrix.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*70}")
    print(f"{model_name} Test Results:")
    print(f"  Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Macro Avg Precision: {report['macro avg']['precision']:.4f}")
    print(f"  Macro Avg Recall: {report['macro avg']['recall']:.4f}")
    print(f"  Macro Avg F1-Score: {report['macro avg']['f1-score']:.4f}")
    print(f"{'='*70}\n")
    
    return accuracy, report

def main():
    print("\n" + "="*70)
    print("SOYBEAN DISEASE CLASSIFICATION - MobileNet & DenseNet Training")
    print("="*70)
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("="*70 + "\n")
    
    # Dataset path
    dataset_path = "data/final_dataset_enhanced"
    if not Path(dataset_path).exists():
        dataset_path = "data/final_dataset"
        if not Path(dataset_path).exists():
            raise FileNotFoundError("Dataset not found!")
    
    print(f"Loading dataset from: {dataset_path}")
    
    # Create datasets with appropriate transforms
    train_transform = create_preprocessing_transforms(is_training=True)
    val_test_transform = create_preprocessing_transforms(is_training=False)
    
    # Load full dataset first
    temp_dataset = SoybeanDiseaseDataset(dataset_path, transform=None)
    class_names = temp_dataset.classes
    num_classes = len(class_names)
    
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names}")
    print(f"Total samples: {len(temp_dataset)}\n")
    
    # Create actual dataset with transforms
    full_dataset = SoybeanDiseaseDataset(dataset_path, transform=val_test_transform)
    
    # Split dataset (70% train, 15% val, 15% test)
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply training augmentation to train set
    train_dataset.dataset.transform = train_transform
    
    print(f"Dataset split:")
    print(f"  Training:   {len(train_dataset)} samples ({len(train_dataset)/total_size*100:.1f}%)")
    print(f"  Validation: {len(val_dataset)} samples ({len(val_dataset)/total_size*100:.1f}%)")
    print(f"  Test:       {len(test_dataset)} samples ({len(test_dataset)/total_size*100:.1f}%)\n")
    
    # Determine batch size based on GPU memory
    if device.type == 'cuda':
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory_gb >= 8:
            batch_size = 16
        elif gpu_memory_gb >= 6:
            batch_size = 12
        else:
            batch_size = 8
    else:
        batch_size = 4
    
    print(f"Batch size: {batch_size}\n")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=2, pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=2, pin_memory=True if device.type == 'cuda' else False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=2, pin_memory=True if device.type == 'cuda' else False)
    
    # Models to train
    models_config = [
        {"name": "MobileNet", "epochs": 50},
        {"name": "DenseNet", "epochs": 40}
    ]
    
    results = {}
    
    for config in models_config:
        model_name = config["name"]
        num_epochs = config["epochs"]
        
        print(f"\n{'#'*70}")
        print(f"# TRAINING {model_name.upper()}")
        print(f"{'#'*70}\n")
        
        # Create model
        model = create_model(model_name, num_classes)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable\n")
        
        # Train model
        trained_model, best_val_acc = train_model(
            model, train_loader, val_loader,
            num_epochs=num_epochs,
            model_name=model_name,
            save_dir="CNN_trained_models"
        )
        
        # Evaluate on test set
        test_acc, test_report = evaluate_model(
            trained_model, test_loader, class_names,
            model_name=model_name,
            save_dir="CNN_trained_models"
        )
        
        results[model_name] = {
            "best_val_accuracy": best_val_acc.item(),
            "test_accuracy": test_acc,
            "test_precision": test_report['macro avg']['precision'],
            "test_recall": test_report['macro avg']['recall'],
            "test_f1": test_report['macro avg']['f1-score']
        }
        
        # Clear GPU memory
        del model, trained_model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        print(f"\n{model_name} Training Complete!\n")
    
    # Save overall results
    results_path = Path("CNN_trained_models") / "mobilenet_densenet_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE - FINAL RESULTS")
    print("="*70)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Best Val Acc:  {metrics['best_val_accuracy']:.4f} ({metrics['best_val_accuracy']*100:.2f}%)")
        print(f"  Test Accuracy: {metrics['test_accuracy']:.4f} ({metrics['test_accuracy']*100:.2f}%)")
        print(f"  Test Precision: {metrics['test_precision']:.4f}")
        print(f"  Test Recall:    {metrics['test_recall']:.4f}")
        print(f"  Test F1-Score:  {metrics['test_f1']:.4f}")
    
    print("\n" + "="*70)
    best_model = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
    print(f"Best model: {best_model} with {results[best_model]['test_accuracy']*100:.2f}% test accuracy")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
