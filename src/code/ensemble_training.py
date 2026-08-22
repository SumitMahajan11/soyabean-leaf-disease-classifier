"""
ADVANCED ENSEMBLE TRAINING FOR SOYBEAN DISEASE CLASSIFICATION
Target: Push accuracy beyond 98.14% using ensemble methods and advanced techniques
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
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

def create_advanced_transforms(train_phase=True):
    """Create advanced preprocessing pipeline with sophisticated data augmentation for training"""
    if train_phase:
        # Advanced transforms with stronger data augmentation for training
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3))
        ])
    else:
        # Standard transforms for validation/test (no augmentation)
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def create_ensemble_model(num_classes):
    """Create an ensemble of different EfficientNet variants"""
    # Create multiple model variants
    model1 = models.efficientnet_b3(pretrained=True)  # Main model
    model1.classifier[1] = nn.Linear(model1.classifier[1].in_features, num_classes)
    
    model2 = models.efficientnet_b2(pretrained=True)  # Secondary model
    model2.classifier[1] = nn.Linear(model2.classifier[1].in_features, num_classes)
    
    model3 = models.efficientnet_b4(pretrained=True)  # Tertiary model
    model3.classifier[1] = nn.Linear(model3.classifier[1].in_features, num_classes)
    
    # Initialize all models
    model1 = model1.to(device)
    model2 = model2.to(device)
    model3 = model3.to(device)
    
    return [model1, model2, model3]

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

def train_ensemble_model(models, train_loader, val_loader, num_epochs=70, model_names=["EffNetB3", "EffNetB2", "EffNetB4"], save_dir="CNN_trained_models"):
    """Train the ensemble models with advanced techniques"""
    save_path = Path(save_dir) / "AdvancedEnsemble"
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate class weights for imbalanced dataset
    class_weights = calculate_class_weights(train_loader.dataset.dataset)
    print(f"Class weights: {class_weights}")
    
    # Create separate optimizers and schedulers for each model
    optimizers = []
    schedulers = []
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    for model in models:
        # Use different learning rates for different models
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6)
        optimizers.append(optimizer)
        schedulers.append(scheduler)
    
    # Mixed precision training if available
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    best_models_wts = [copy.deepcopy(model.state_dict()) for model in models]
    best_acc = 0.0
    best_epoch = 0
    
    train_losses = [[] for _ in models]
    train_accuracies = [[] for _ in models]
    val_losses = [[] for _ in models]
    val_accuracies = [[] for _ in models]
    
    print(f"Starting advanced ensemble training...")
    print(f"Using class weights to handle imbalanced dataset")
    print(f"Using AdamW optimizer with CosineAnnealingWarmRestarts scheduler")
    print(f"Using ensemble of EfficientNet-B2, B3, and B4")
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        for i, model in enumerate(models):
            model.train()
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizers[i].zero_grad()
                
                # Mixed precision training
                with torch.cuda.amp.autocast() if scaler else torch.no_grad():
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                
                if scaler:
                    scaler.scale(loss).backward()
                    # Gradient clipping to prevent exploding gradients
                    scaler.unscale_(optimizers[i])
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizers[i])
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizers[i].step()
                
                # Update learning rate scheduler
                schedulers[i].step(epoch + len(train_loader) / len(train_loader.dataset))
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_train_loss = running_loss / len(train_loader.dataset)
            epoch_train_acc = running_corrects.double() / len(train_loader.dataset)
            
            train_losses[i].append(epoch_train_loss)
            train_accuracies[i].append(epoch_train_acc.item())
        
        # Print training metrics for each model
        for i, name in enumerate(model_names):
            print(f'{name} Train Loss: {train_losses[i][-1]:.4f} Acc: {train_accuracies[i][-1]:.4f}')
        
        # Validation phase - evaluate ensemble performance
        ensemble_val_acc = evaluate_ensemble(models, val_loader)
        print(f'Ensemble Val Acc: {ensemble_val_acc:.4f}')
        
        # Deep copy the best ensemble
        if ensemble_val_acc > best_acc:
            best_acc = ensemble_val_acc
            best_models_wts = [copy.deepcopy(model.state_dict()) for model in models]
            best_epoch = epoch + 1
            # Save the best ensemble
            ensemble_checkpoint = {
                'epoch': epoch + 1,
                'models_state_dict': [model.state_dict() for model in models],
                'optimizers_state_dict': [opt.state_dict() for opt in optimizers],
                'best_acc': best_acc,
                'val_acc': ensemble_val_acc,
            }
            torch.save(ensemble_checkpoint, save_path / "best_ensemble_checkpoint.pth")
            print(f'✓ New best ensemble saved! Best Val Acc: {best_acc:.4f}')
        
        print(f'Best ensemble val Acc: {best_acc:.4f} at epoch {best_epoch}')
        print()
    
    print(f'Best ensemble val Acc: {best_acc:.4f} at epoch {best_epoch}')
    
    # Load best ensemble weights
    for i, model in enumerate(models):
        model.load_state_dict(best_models_wts[i])
    
    # Save the best ensemble models individually too
    for i, model in enumerate(models):
        torch.save(model.state_dict(), save_path / f"model_{model_names[i]}.pth")
    
    # Save training metrics
    metrics = {
        "model_names": model_names,
        "best_accuracy": best_acc.item(),
        "best_epoch": best_epoch,
        "num_epochs": num_epochs,
        "train_losses": {name: losses for name, losses in zip(model_names, train_losses)},
        "train_accuracies": {name: accs for name, accs in zip(model_names, train_accuracies)},
        "val_losses": {name: losses for name, losses in zip(model_names, val_losses)},
        "val_accuracies": {name: accs for name, accs in zip(model_names, val_accuracies)},
    }
    
    with open(save_path / "ensemble_metrics.json", 'w') as f:
        import json
        json.dump(metrics, f, indent=4)
    
    # Save training log
    with open(save_path / "ensemble_training_log.txt", 'w') as f:
        f.write(f"Advanced Ensemble Model Training\n")
        f.write(f"Best Accuracy: {best_acc:.4f}\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Training completed at: {datetime.now()}\n")
        f.write(f"Used ensemble of EfficientNet-B2, B3, and B4\n")
        f.write(f"Used AdamW optimizer\n")
        f.write(f"Used CosineAnnealingWarmRestarts scheduler\n")
        f.write(f"Used class weights for imbalanced dataset\n")
        f.write(f"Used mixed precision training\n")
        f.write(f"Used gradient clipping\n")
        f.write(f"Used advanced data augmentation\n")
    
    # Plot training curves for each model
    plt.figure(figsize=(15, 10))
    
    for i, name in enumerate(model_names):
        plt.subplot(2, 2, i+1)
        plt.plot(train_accuracies[i], label=f'{name} Train')
        plt.plot(val_accuracies[i], label=f'{name} Val')
        plt.title(f'{name} - Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path / "ensemble_training_curves.png")
    plt.close()
    
    return models, best_acc

def evaluate_ensemble(models, val_loader):
    """Evaluate the ensemble performance"""
    for model in models:
        model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Get predictions from all models and average them
            ensemble_outputs = 0
            for model in models:
                outputs = model(inputs)
                ensemble_outputs += torch.softmax(outputs, dim=1)
            
            # Average the predictions
            ensemble_outputs /= len(models)
            _, ensemble_preds = torch.max(ensemble_outputs, 1)
            
            all_preds.extend(ensemble_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

def evaluate_single_model(model, test_loader):
    """Evaluate a single model"""
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
    
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

def main():
    print("Starting Advanced Ensemble Soybean Disease Classification Training")
    print(f"Using device: {device}")
    
    # Create advanced preprocessing transforms
    train_transform = create_advanced_transforms(train_phase=True)
    val_transform = create_advanced_transforms(train_phase=False)
    
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
    batch_size = 4 if device.type == 'cuda' else 2  # Reduced batch size for ensemble
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True if device.type == 'cuda' else False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True if device.type == 'cuda' else False)
    
    # Create ensemble models
    print("\nCreating Ensemble Models (EfficientNet-B2, B3, B4)...")
    models = create_ensemble_model(num_classes)
    
    # Train the ensemble model
    print(f"\n{'='*60}")
    print(f"Training Advanced Ensemble Model")
    print(f"{'='*60}")
    
    trained_models, best_acc = train_ensemble_model(
        models, train_loader, val_loader, 
        num_epochs=70,  # More epochs with better techniques
        model_names=["EffNetB2", "EffNetB3", "EffNetB4"],
        save_dir="CNN_trained_models"
    )
    
    # Evaluate each individual model on test set
    print("\nEvaluating individual models on test set:")
    for i, (model, name) in enumerate(zip(trained_models, ["EffNetB2", "EffNetB3", "EffNetB4"])):
        test_acc = evaluate_single_model(model, test_loader)
        print(f"  {name} Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Evaluate ensemble on test set
    print("\nEvaluating ensemble on test set:")
    ensemble_test_acc = evaluate_ensemble(trained_models, test_loader)
    print(f"  Ensemble Test Accuracy: {ensemble_test_acc:.4f} ({ensemble_test_acc*100:.2f}%)")
    
    print(f"\nAdvanced Ensemble completed - Val Acc: {best_acc:.4f}, Test Acc: {ensemble_test_acc:.4f}")
    
    print("\n" + "="*60)
    print("Advanced ensemble training completed!")
    print(f"Final Ensemble Test Accuracy: {ensemble_test_acc:.4f} ({ensemble_test_acc*100:.2f}%)")
    print("Models saved to CNN_trained_models/AdvancedEnsemble/")
    print("="*60)

if __name__ == "__main__":
    main()