import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.transforms import functional as TF
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
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))  # Store as string path
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image - use context manager to ensure file is properly closed
        try:
            with Image.open(img_path) as img:
                image = img.convert('RGB').copy()  # Copy to ensure it's in memory and can be closed
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image if there's an error
            image = Image.new('RGB', (512, 512), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_preprocessing_transforms():
    """Create the unified preprocessing pipeline as specified"""
    return transforms.Compose([
        transforms.Resize((512, 512)),  # Mandatory 512x512 resize
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

def create_model(model_name, num_classes):
    """Create and initialize the specified model"""
    if model_name.lower() == 'mobilenet':
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name.lower() == 'densenet':
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model.to(device)

def train_model(model, train_loader, val_loader, num_epochs=10, model_name="model", save_dir="CNN_trained_models"):
    """Train a single model with early stopping and learning rate scheduling - Minimal Memory Version"""
    save_path = Path(save_dir) / model_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Mixed precision training if available
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    print(f"Starting training for {model_name}...")
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            with torch.cuda.amp.autocast() if scaler else torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            
            # Backward pass
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # Clear cache periodically to manage memory
            if batch_idx % 20 == 0:  # More frequent clearing
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
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
        
        # Learning rate scheduling
        scheduler.step(epoch_val_acc)
        
        print(f'Best val Acc: {best_acc:.4f} at epoch {best_epoch}')
        print()
    
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
        "num_epochs": num_epochs,
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
        f.write(f"Best Accuracy: {best_acc:.4f}\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Training completed at: {datetime.now()}\n")
    
    # Plot training curves - minimal version
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
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
    plt.savefig(save_path / "training_curves.png", dpi=150)  # Lower dpi for smaller file
    plt.close()
    
    return model, best_acc

def evaluate_model(model, test_loader, class_names, model_name="model", save_dir="CNN_trained_models"):
    """Evaluate the model and generate detailed metrics - Minimal Memory Version"""
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
    
    # Plot confusion matrix - minimal version
    plt.figure(figsize=(8, 6))  # Smaller figure
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path / "confusion_matrix.png", dpi=150, bbox_inches='tight')  # Lower dpi
    plt.close()
    
    print(f"{model_name} - Accuracy: {accuracy:.4f}")
    return accuracy

def main():
    print("Starting training for remaining models: MobileNet and DenseNet (Minimal Memory Version)")
    print(f"Using device: {device}")
    
    # Create preprocessing transforms
    transform = create_preprocessing_transforms()
    
    # Load dataset
    dataset_path = "data/final_dataset_enhanced"
    if not Path(dataset_path).exists():
        dataset_path = "data/final_dataset"
        if not Path(dataset_path).exists():
            raise FileNotFoundError("Dataset not found in either 'data/final_dataset_enhanced' or 'data/final_dataset'")
    
    print(f"Loading dataset from: {dataset_path}")
    
    # Create full dataset
    full_dataset = SoybeanDiseaseDataset(dataset_path, transform=transform)
    
    # Get class names
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names}")
    
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
    
    # Create data loaders with minimal memory footprint
    batch_size = 2 if device.type == 'cuda' else 1  # Minimal batch size for memory efficiency
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)  # num_workers=0 to reduce memory
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    # Define models to train (only the remaining ones)
    models_to_train = ["MobileNet", "DenseNet"]
    
    # Load existing results if they exist
    results_path = Path("CNN_trained_models") / "overall_results.json"
    results = {}
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
    
    # Train each remaining model
    for model_name in models_to_train:
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        
        # Create model
        model = create_model(model_name.lower(), num_classes)
        
        # Train model
        trained_model, best_acc = train_model(
            model, train_loader, val_loader, 
            num_epochs=5,  # Minimal epochs for memory management
            model_name=model_name,
            save_dir="CNN_trained_models"
        )
        
        # Evaluate model
        test_acc = evaluate_model(
            trained_model, test_loader, class_names,
            model_name=model_name,
            save_dir="CNN_trained_models"
        )
        
        results[model_name] = {
            "best_val_accuracy": best_acc.item(),
            "test_accuracy": test_acc
        }
        
        print(f"{model_name} completed - Val Acc: {best_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        # Clear GPU cache after each model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Save updated overall results
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n" + "="*50)
    print("Training completed for all remaining models!")
    print("Results:")
    for model_name, metrics in results.items():
        print(f"  {model_name}: Val Acc: {metrics['best_val_accuracy']:.4f}, Test Acc: {metrics['test_accuracy']:.4f}")
    print("="*50)
    
    # Identify best performing model
    best_model = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
    print(f"\nBest performing model: {best_model} with test accuracy: {results[best_model]['test_accuracy']:.4f}")

if __name__ == "__main__":
    main()