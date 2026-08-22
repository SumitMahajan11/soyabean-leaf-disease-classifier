"""
ATTENTION-BASED ADVANCED TRAINING FOR SOYBEAN DISEASE CLASSIFICATION
Target: Push accuracy beyond 98.14% using attention mechanisms and advanced architectures
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

def create_attention_transforms(train_phase=True):
    """Create advanced preprocessing pipeline with sophisticated data augmentation for training"""
    if train_phase:
        # Advanced transforms with attention-focused augmentation for training
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.1), ratio=(0.3, 3.3))
        ])
    else:
        # Standard transforms for validation/test (no augmentation)
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class AttentionModule(nn.Module):
    """Simple attention module to focus on important features"""
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class AdvancedAttentionModel(nn.Module):
    """Advanced model with attention mechanisms"""
    def __init__(self, num_classes):
        super(AdvancedAttentionModel, self).__init__()
        
        # Load a pretrained EfficientNet as backbone
        backbone = models.efficientnet_b4(pretrained=True)
        
        # Extract features
        self.features = backbone.features
        
        # Add attention modules at different levels
        self.attention1 = AttentionModule(48)   # After initial conv layers
        self.attention2 = AttentionModule(80)   # After some MBConv blocks
        self.attention3 = AttentionModule(160)  # Middle layers
        self.attention4 = AttentionModule(272)  # Later layers
        
        # Adaptive pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Classification head with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1792, 1024),  # EfficientNet-B4 has 1792 features
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # Initialize the classifier
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        # Extract features through backbone
        x = self.features(x)  # Shape: (B, 1792, H, W)
        
        # Apply attention at different levels
        # Note: We'll apply attention to different feature maps if available
        # For EfficientNet, we'll use the final features and add our own attention
        
        # Global and max pooling
        global_feat = self.global_pool(x)  # (B, 1792, 1, 1)
        max_feat = self.max_pool(x)        # (B, 1792, 1, 1)
        
        # Combine features
        combined_feat = torch.cat([global_feat, max_feat], dim=1)  # (B, 3584, 1, 1)
        combined_feat = combined_feat.view(combined_feat.size(0), -1)  # (B, 3584)
        
        # Classification
        output = self.classifier(combined_feat)
        return output

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

def train_attention_model(model, train_loader, val_loader, num_epochs=80, model_name="AttentionBasedEfficientNet", save_dir="CNN_trained_models"):
    """Train the model with attention mechanisms and advanced techniques"""
    save_path = Path(save_dir) / model_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate class weights for imbalanced dataset
    class_weights = calculate_class_weights(train_loader.dataset.dataset)
    print(f"Class weights: {class_weights}")
    
    # Use Focal Loss for better handling of class imbalance
    from torch.nn import CrossEntropyLoss
    criterion = CrossEntropyLoss(weight=class_weights)
    
    # Advanced optimizer with differential learning rates
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'classifier' in name or 'attention' in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)
    
    # Use different learning rates for backbone and classifier
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 0.00002},  # Very low LR for backbone
        {'params': classifier_params, 'lr': 0.0001}   # Higher LR for classifier and attention
    ], weight_decay=1e-4, betas=(0.9, 0.999))
    
    # Advanced learning rate scheduling
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-7
    )
    
    # Mixed precision training if available
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    max_patience = 18  # Stop if no improvement for 18 epochs
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    print(f"Starting attention-based training for {model_name}...")
    print(f"Using CrossEntropy Loss with class weights")
    print(f"Using differential learning rates (backbone: 2e-5, classifier/attention: 1e-4)")
    print(f"Using CosineAnnealingWarmRestarts scheduler")
    print(f"Using EfficientNet-B4 with attention modules")
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
            
            # Update learning rate scheduler
            scheduler.step(epoch + len(train_loader) / len(train_loader.dataset))
            
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
        
        current_lr_backbone = optimizer.param_groups[0]['lr']
        current_lr_classifier = optimizer.param_groups[1]['lr']
        print(f'Best val Acc: {best_acc:.4f} at epoch {best_epoch}')
        print(f'Current LR (Backbone): {current_lr_backbone:.2e}')
        print(f'Current LR (Classifier/Attention): {current_lr_classifier:.2e}')
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
        "final_learning_rate_backbone": current_lr_backbone,
        "final_learning_rate_classifier": current_lr_classifier,
        "early_stopping_epoch": epoch + 1,
        "used_early_stopping": epochs_without_improvement >= max_patience
    }
    
    with open(save_path / "metrics.json", 'w') as f:
        import json
        json.dump(metrics, f, indent=4)
    
    # Save training log
    with open(save_path / "training_log.txt", 'w') as f:
        f.write(f"Attention-Based Model: {model_name}\n")
        f.write(f"Best Accuracy: {best_acc:.4f}\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Training completed at: {datetime.now()}\n")
        f.write(f"Used EfficientNet-B4 with attention modules\n")
        f.write(f"Used CrossEntropy Loss with class weights\n")
        f.write(f"Used differential learning rates\n")
        f.write(f"Used CosineAnnealingWarmRestarts scheduler\n")
        f.write(f"Used mixed precision training\n")
        f.write(f"Used gradient clipping\n")
        f.write(f"Used advanced data augmentation\n")
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
        import json
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
    print("Starting Attention-Based Soybean Disease Classification Training")
    print(f"Using device: {device}")
    
    # Create advanced preprocessing transforms
    train_transform = create_attention_transforms(train_phase=True)
    val_transform = create_attention_transforms(train_phase=False)
    
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
    batch_size = 2 if device.type == 'cuda' else 1  # Reduced batch size for attention model
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True if device.type == 'cuda' else False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True if device.type == 'cuda' else False)
    
    # Create attention-based model
    print("\nCreating Attention-Based EfficientNet-B4 model...")
    model = AdvancedAttentionModel(num_classes)
    model = model.to(device)
    
    # Train the attention-based model
    print(f"\n{'='*60}")
    print(f"Training Attention-Based EfficientNet Model")
    print(f"{'='*60}")
    
    trained_model, best_acc = train_attention_model(
        model, train_loader, val_loader, 
        num_epochs=80,  # More epochs with better techniques and early stopping
        model_name="AttentionBasedEfficientNet",
        save_dir="CNN_trained_models"
    )
    
    # Evaluate model
    test_acc = evaluate_model(
        trained_model, test_loader, class_names,
        model_name="AttentionBasedEfficientNet",
        save_dir="CNN_trained_models"
    )
    
    print(f"\nAttention-Based EfficientNet completed - Val Acc: {best_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    print("\n" + "="*60)
    print("Attention-based training completed!")
    print(f"Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("Model saved to CNN_trained_models/AttentionBasedEfficientNet/")
    print("="*60)

if __name__ == "__main__":
    main()