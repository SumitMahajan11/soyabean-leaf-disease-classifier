"""
KNOWLEDGE DISTILLATION APPROACH FOR SOYBEAN DISEASE CLASSIFICATION
Target: Create a high-performance student model that potentially exceeds the teacher model
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
from sklearn.metrics import accuracy_score
from datetime import datetime
import copy
import warnings
warnings.filterwarnings('ignore')

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
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

def create_distillation_transforms(train_phase=True):
    """Create transforms for knowledge distillation"""
    if train_phase:
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.05), ratio=(0.3, 3.3))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class StudentModel(nn.Module):
    """Lighter but effective student model for knowledge distillation"""
    def __init__(self, num_classes, teacher_features=1000):
        super(StudentModel, self).__init__()
        
        # Use EfficientNet-B2 as student (smaller than B3 but still powerful)
        backbone = models.efficientnet_b2(pretrained=True)
        
        # Extract features
        self.features = backbone.features
        
        # Adaptive pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1408, 768),  # EfficientNet-B2 has 1408 features
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(768, num_classes)
        )
        
        # Initialize the classifier
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def load_teacher_model(num_classes):
    """Load our best performing teacher model (Enhanced EfficientNet-B3)"""
    teacher = models.efficientnet_b3(pretrained=False)
    teacher.classifier[1] = nn.Linear(teacher.classifier[1].in_features, num_classes)
    
    # Load the best performing model
    checkpoint_path = "CNN_trained_models/EnhancedEfficientNet/best_model_checkpoint.pth"
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        teacher.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        print("✓ Teacher model (Enhanced EfficientNet-B3) loaded")
    else:
        # Fallback to original Enhanced EfficientNet
        model_path = "CNN_trained_models/EnhancedEfficientNet/model.pth"
        if Path(model_path).exists():
            teacher.load_state_dict(torch.load(model_path, map_location=device))
            print("✓ Teacher model loaded from model.pth")
        else:
            print("⚠ Teacher model not found, using pretrained weights only")
    
    teacher = teacher.to(device)
    teacher.eval()  # Set to eval mode
    return teacher

def distillation_loss(student_outputs, teacher_outputs, labels, T=4.0, alpha=0.7):
    """
    Compute knowledge distillation loss
    L = alpha * T^2 * KL(student_logits/T, teacher_logits/T) + (1-alpha) * CE(student_logits, labels)
    """
    # Soft targets from teacher
    soft_targets = torch.softmax(teacher_outputs / T, dim=1)
    soft_prob = torch.log_softmax(student_outputs / T, dim=1)
    distill_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size(0)
    
    # Student's loss on true labels
    student_loss = nn.CrossEntropyLoss()(student_outputs, labels)
    
    # Combined loss
    total_loss = alpha * T * T * distill_loss + (1 - alpha) * student_loss
    return total_loss, student_loss, distill_loss

def train_knowledge_distillation(teacher_model, student_model, train_loader, val_loader, num_epochs=60, save_dir="CNN_trained_models"):
    """Train student model using knowledge distillation"""
    save_path = Path(save_dir) / "KnowledgeDistilledModel"
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Loss function and optimizer
    optimizer = optim.AdamW(student_model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Mixed precision training if available
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    best_model_wts = copy.deepcopy(student_model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    
    train_losses = []
    val_accuracies = []
    
    print(f"Starting knowledge distillation training...")
    print(f"Teacher model: Enhanced EfficientNet-B3 (98.14%)")
    print(f"Student model: EfficientNet-B2 with distillation")
    print(f"Using temperature T=4.0 and alpha=0.7")
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        student_model.train()
        teacher_model.eval()  # Teacher is frozen
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass through both models
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)
            
            with torch.cuda.amp.autocast() if scaler else torch.no_grad():
                student_outputs = student_model(inputs)
                loss, student_loss, distill_loss = distillation_loss(student_outputs, teacher_outputs, labels)
            
            if scaler:
                scaler.scale(loss).backward()
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step(epoch + len(train_loader) / len(train_loader.dataset))
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(student_outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = running_corrects.double() / len(train_loader.dataset)
        
        train_losses.append(epoch_train_loss)
        
        print(f'Train Loss: {epoch_train_loss:.4f}, Distill Loss: {distill_loss.item():.4f}, Student Loss: {student_loss.item():.4f}')
        
        # Validation phase
        student_model.eval()
        running_corrects = 0
        
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast() if scaler else torch.no_grad():
                    outputs = student_model(inputs)
                
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
        
        epoch_val_acc = running_corrects.double() / len(val_loader.dataset)
        val_accuracies.append(epoch_val_acc.item())
        
        print(f'Val Acc: {epoch_val_acc:.4f}')
        
        # Deep copy the best model
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(student_model.state_dict())
            best_epoch = epoch + 1
            # Save the best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'val_acc': epoch_val_acc,
            }, save_path / "best_model_checkpoint.pth")
            print(f'✓ New best model saved! Best Val Acc: {best_acc:.4f}')
        
        print(f'Best val Acc: {best_acc:.4f} at epoch {best_epoch}')
        print()
    
    print(f'Best val Acc: {best_acc:.4f} at epoch {best_epoch}')
    
    # Load best model weights
    student_model.load_state_dict(best_model_wts)
    
    # Save the best model
    torch.save(student_model.state_dict(), save_path / "model.pth")
    
    # Save training metrics
    metrics = {
        "model_name": "KnowledgeDistilledModel",
        "teacher_model": "Enhanced EfficientNet-B3 (98.14%)",
        "student_model": "EfficientNet-B2 with distillation",
        "best_accuracy": best_acc.item(),
        "best_epoch": best_epoch,
        "num_epochs": num_epochs,
        "train_losses": train_losses,
        "val_accuracies": val_accuracies,
    }
    
    with open(save_path / "metrics.json", 'w') as f:
        import json
        json.dump(metrics, f, indent=4)
    
    # Plot training curves
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.axhline(y=best_acc.item(), color='r', linestyle='--', label=f'Best Val Acc: {best_acc:.4f}')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path / "training_curves.png")
    plt.close()
    
    return student_model, best_acc

def evaluate_model(model, test_loader, model_name="model", save_dir="CNN_trained_models"):
    """Evaluate the model"""
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
    
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"{model_name} - Accuracy: {accuracy:.4f}")
    return accuracy

def main():
    print("Starting Knowledge Distillation Training")
    print(f"Using device: {device}")
    
    # Create transforms
    train_transform = create_distillation_transforms(train_phase=True)
    val_transform = create_distillation_transforms(train_phase=False)
    
    # Load dataset
    dataset_path = "data/final_dataset_enhanced"
    if not Path(dataset_path).exists():
        dataset_path = "data/final_dataset"
        if not Path(dataset_path).exists():
            raise FileNotFoundError("Dataset not found in either 'data/final_dataset_enhanced' or 'data/final_dataset'")
    
    print(f"Loading dataset from: {dataset_path}")
    
    # Create full dataset
    full_dataset = SoybeanDiseaseDataset(dataset_path, transform=None)
    
    # Get class names
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Dataset split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform
    
    # Create data loaders
    batch_size = 4 if device.type == 'cuda' else 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True if device.type == 'cuda' else False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True if device.type == 'cuda' else False)
    
    # Load teacher model (our best performer)
    print("\nLoading teacher model...")
    teacher_model = load_teacher_model(num_classes)
    
    # Create student model
    print("\nCreating student model...")
    student_model = StudentModel(num_classes)
    student_model = student_model.to(device)
    
    # Train using knowledge distillation
    print(f"\n{'='*60}")
    print(f"Training Student Model via Knowledge Distillation")
    print(f"{'='*60}")
    
    trained_student, best_acc = train_knowledge_distillation(
        teacher_model, student_model, train_loader, val_loader,
        num_epochs=60,
        save_dir="CNN_trained_models"
    )
    
    # Evaluate the distilled model
    test_acc = evaluate_model(
        trained_student, test_loader,
        model_name="KnowledgeDistilledModel",
        save_dir="CNN_trained_models"
    )
    
    print(f"\nKnowledge Distillation completed!")
    print(f"Student Model Val Acc: {best_acc:.4f}")
    print(f"Student Model Test Acc: {test_acc:.4f}")
    print(f"Teacher Model Acc: 98.14%")
    
    print("\n" + "="*60)
    print("Knowledge distillation training completed!")
    print(f"Student Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("Model saved to CNN_trained_models/KnowledgeDistilledModel/")
    print("="*60)

if __name__ == "__main__":
    main()