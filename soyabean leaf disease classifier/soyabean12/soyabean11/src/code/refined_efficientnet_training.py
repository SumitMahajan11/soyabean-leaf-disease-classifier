"""
REFINED EFFICIENTNET TRAINING FOR SOYBEAN DISEASE CLASSIFICATION
Target: 96-98% accuracy improvement over baseline 94.43%
"""

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

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

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

def create_refined_transforms(train_phase=True):
    """Create refined preprocessing pipeline with controlled data augmentation for training"""
    if train_phase:
        # Refined transforms with more conservative data augmentation for training
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.05), ratio=(0.3, 3.3))
        ])
    else:
        # Standard transforms for validation/test (no augmentation)
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def create_refined_model(num_classes):
    """Create a refined EfficientNet model with proper initialization"""
    # Use EfficientNet-B1 for better performance than B0 but less memory than B3
    model = models.efficientnet_b1(pretrained=True)  # Using B1 for better balance
    # Update the classifier head
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    # Initialize the new classifier layer properly
    nn.init.xavier_uniform_(model.classifier[1].weight)
    nn.init.constant_(model.classifier[1].bias, 0)
    
    return model.to(device)

def calculate_class_weights(dataset):
    """Calculate class weights to handle imbalanced dataset"""
    # Count samples per class
    class_counts = [0] * len(dataset.classes)
    for _, label in dataset.samples:
        class_counts[label] += 1
    
    # Calculate weights (inverse frequency with square root for less aggressive weighting)
    total_samples = len(dataset.samples)
    class_weights = []
    for count in class_counts:
        weight = np.sqrt(total_samples / (len(class_counts) * count))
        class_weights.append(weight)
    
    return torch.tensor(class_weights, dtype=torch.float).to(device)

def train_refined_model(model, train_loader, val_loader, num_epochs=80, model_name="RefinedEfficientNet", save_dir="CNN_trained_models"):
    """Train the model with refined techniques"""
    save_path = Path(save_dir) / model_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate class weights for imbalanced dataset
    class_weights = calculate_class_weights(train_loader.dataset.dataset)
    print(f"Class weights: {class_weights}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # Use a conservative learning rate with a good optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5, betas=(0.9, 0.999))
    
    # More conservative learning rate scheduling
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7, min_lr=1e-7)
    
    # Mixed precision training if available
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    max_patience = 15  # Stop if no improvement for 15 epochs
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    print(f"Starting refined training for {model_name}...")
    print(f"Using class weights to handle imbalanced dataset")
    print(f"Using AdamW optimizer with ReduceLROnPlateau scheduler")
    print(f"Using EfficientNet-B1 (balanced model)")
    print(f"Early stopping with patience of {max_patience}")
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision training
            with torch.cuda.amp.autocast() if scaler else torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            
            if scaler:
                scaler.scale(loss).backward()
                # Gradient clipping to prevent exploding gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = running_corrects.double() / len(train_loader.dataset)
        
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc.item())
        
        print(f'Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}')
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast() if scaler else torch.no_grad():
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        
        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = running_corrects.double() / len(val_loader.dataset)
        
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc.item())
        
        print(f'Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}')
        
        # Deep copy the best model
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_without_improvement = 0  # Reset counter
            # Save the best model checkpoint
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'val_acc': epoch_val_acc,
            }, save_path / "best_model_checkpoint.pth")
            print(f'✓ New best model saved! Best Val Acc: {best_acc:.4f}')
        else:
            epochs_without_improvement += 1
        
        # Learning rate scheduling based on validation accuracy
        scheduler.step(epoch_val_acc)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Best val Acc: {best_acc:.4f} at epoch {best_epoch}')
        print(f'Current LR: {current_lr:.2e}')
        print(f'No improvement for: {epochs_without_improvement}/{max_patience} epochs')
        print()
        
        # Early stopping
        if epochs_without_improvement >= max_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    print(f'Best val Acc: {best_acc:.4f} at epoch {best_epoch}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Save the best model
    torch.save(model.state_dict(), save_path / "model.pth")
    
    # Save training metrics
    metrics = {
        "model_name": model_name,
        "best_accuracy": best_acc.item(),
        "best_epoch": best_epoch,
        "num_epochs": epoch + 1,  # Actual epochs run (due to early stopping)
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "final_learning_rate": current_lr,
        "early_stopping_epoch": epoch + 1,
        "used_early_stopping": epochs_without_improvement >= max_patience
    }
    
    with open(save_path / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save training log
    with open(save_path / "training_log.txt", 'w') as f:
        f.write(f"Refined Model: {model_name}\n")
        f.write(f"Best Accuracy: {best_acc:.4f}\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Training completed at: {datetime.now()}\n")
        f.write(f"Used EfficientNet-B1\n")
        f.write(f"Used AdamW optimizer\n")
        f.write(f"Used ReduceLROnPlateau scheduler\n")
        f.write(f"Used class weights for imbalanced dataset\n")
        f.write(f"Used mixed precision training\n")
        f.write(f"Used gradient clipping\n")
        f.write(f"Used conservative data augmentation\n")
        f.write(f"Early stopping triggered: {epochs_without_improvement >= max_patience}\n")
        f.write(f"Stopped at epoch: {epoch + 1}\n")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.axhline(y=best_acc.item(), color='r', linestyle='--', label=f'Best Val Acc: {best_acc:.4f}')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path / "training_curves.png")
    plt.close()
    
    return model, best_acc

def evaluate_model(model, test_loader, class_names, model_name="model", save_dir="CNN_trained_models"):
    """Evaluate the model and generate detailed metrics"""
    save_path = Path(save_dir) / model_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    model.eval()
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
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Save detailed metrics
    metrics = {
        "model_name": model_name,
        "overall_accuracy": accuracy,
        "classification_report": report
    }
    
    with open(save_path / "detailed_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"{model_name} - Accuracy: {accuracy:.4f}")
    return accuracy

def main():
    print("Starting Refined Soybean Disease Classification Training Pipeline")
    print(f"Using device: {device}")
    
    # Create refined preprocessing transforms
    train_transform = create_refined_transforms(train_phase=True)
    val_transform = create_refined_transforms(train_phase=False)
    
    # Load dataset
    dataset_path = "data/final_dataset_enhanced"
    if not Path(dataset_path).exists():
        dataset_path = "data/final_dataset"
        if not Path(dataset_path).exists():
            raise FileNotFoundError("Dataset not found in either 'data/final_dataset_enhanced' or 'data/final_dataset'")
    
    print(f"Loading dataset from: {dataset_path}")
    
    # Create full dataset with training transforms
    full_dataset = SoybeanDiseaseDataset(dataset_path, transform=None)  # We'll apply transforms in the data loaders
    
    # Get class names
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names}")
    
    # Count samples per class to understand imbalance
    class_counts = [0] * num_classes
    for _, label in full_dataset.samples:
        class_counts[label] += 1
    
    print("Class distribution:")
    for i, (class_name, count) in enumerate(zip(class_names, class_counts)):
        print(f"  {class_name}: {count} samples ({count/len(full_dataset)*100:.1f}%)")
    
    # Split dataset into train, validation, test (70%, 15%, 15%)
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Dataset split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Apply transforms to the split datasets
    # Create datasets with specific transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform
    
    # Create data loaders
    batch_size = 6 if device.type == 'cuda' else 2  # Adjusted batch size for B1 model
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True if device.type == 'cuda' else False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True if device.type == 'cuda' else False)
    
    # Create refined model
    print("\nCreating Refined EfficientNet-B1 model...")
    model = create_refined_model(num_classes)
    
    # Train the refined model
    print(f"\n{'='*60}")
    print(f"Training Refined EfficientNet Model")
    print(f"{'='*60}")
    
    trained_model, best_acc = train_refined_model(
        model, train_loader, val_loader, 
        num_epochs=80,  # More epochs with better techniques and early stopping
        model_name="RefinedEfficientNet",
        save_dir="CNN_trained_models"
    )
    
    # Evaluate model
    test_acc = evaluate_model(
        trained_model, test_loader, class_names,
        model_name="RefinedEfficientNet",
        save_dir="CNN_trained_models"
    )
    
    print(f"\nRefined EfficientNet completed - Val Acc: {best_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    print("\n" + "="*60)
    print("Refined training completed!")
    print(f"Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("Model saved to CNN_trained_models/RefinedEfficientNet/")
    print("="*60)

if __name__ == "__main__":
    main()