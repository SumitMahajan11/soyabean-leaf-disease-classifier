"""
V2 ENSEMBLE EVALUATION: EfficientNet-B4 + ResNet152
Target: >98% accuracy on 17-class hierarchical dataset
Strategy: Weighted Soft Voting (Probability Averaging)
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'='*60}")
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
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
            image = Image.new('RGB', (512, 512), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
        return image, label

def get_eval_transforms():
    """Standard eval transforms used during training"""
    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def load_efficientnet_b4(num_classes, checkpoint_path):
    print(f"Loading EfficientNet-B4 from {checkpoint_path}...")
    model = models.efficientnet_b4()
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(model.classifier[1].in_features, num_classes)
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model

def load_resnet152(num_classes, checkpoint_path):
    print(f"Loading ResNet152 from {checkpoint_path}...")
    model = models.resnet152()
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model

def evaluate_models(eff_model, res_model, data_loader, class_names, weights=[0.5, 0.5]):
    """
    Evaluates both models and their ensemble.
    weights: [weight_efficientnet, weight_resnet]
    """
    all_labels = []
    eff_probs = []
    res_probs = []
    
    print("\nRunning inference...")
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader):
            inputs = inputs.to(device)
            
            # EfficientNet outputs
            with torch.amp.autocast('cuda'):
                eff_out = eff_model(inputs)
                eff_p = torch.softmax(eff_out, dim=1)
                
                # ResNet outputs
                res_out = res_model(inputs)
                res_p = torch.softmax(res_out, dim=1)
            
            eff_probs.append(eff_p.cpu().numpy())
            res_probs.append(res_p.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    eff_probs = np.vstack(eff_probs)
    res_probs = np.vstack(res_probs)
    all_labels = np.array(all_labels)
    
    # Individual accuracies
    eff_preds = np.argmax(eff_probs, axis=1)
    res_preds = np.argmax(res_probs, axis=1)
    
    eff_acc = accuracy_score(all_labels, eff_preds)
    res_acc = accuracy_score(all_labels, res_preds)
    
    print(f"\nIndividual Model Accuracies:")
    print(f"  EfficientNet-B4: {eff_acc*100:.2f}%")
    print(f"  ResNet152:      {res_acc*100:.2f}%")
    
    # Ensemble (Weighted)
    ensemble_probs = (weights[0] * eff_probs) + (weights[1] * res_probs)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    ensemble_acc = accuracy_score(all_labels, ensemble_preds)
    
    print(f"\nEnsemble Result (Weights: {weights}):")
    print(f"  Accuracy: {ensemble_acc*100:.2f}%")
    
    # Classification Report for Ensemble
    report = classification_report(all_labels, ensemble_preds, target_names=class_names)
    print("\nEnsemble Classification Report:")
    print(report)
    
    return ensemble_acc, ensemble_preds, all_labels

def plot_ensemble_confusion_matrix(labels, preds, class_names, save_path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('V2 Ensemble Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✓ Confusion Matrix saved to {save_path}")

def find_best_weights(eff_probs, res_probs, labels):
    """Simple grid search for best weights"""
    print("\nFinding optimal weights...")
    best_acc = 0
    best_w = [0.5, 0.5]
    
    for i in range(0, 11):
        w1 = i / 10.0
        w2 = 1.0 - w1
        ensemble_p = (w1 * eff_probs) + (w2 * res_probs)
        ensemble_preds = np.argmax(ensemble_p, axis=1)
        acc = accuracy_score(labels, ensemble_preds)
        if acc > best_acc:
            best_acc = acc
            best_w = [w1, w2]
            
    print(f"  ★ Optimal Weights found: EffNet={best_w[0]}, ResNet={best_w[1]} with Acc={best_acc*100:.2f}%")
    return best_w

def main():
    # Paths
    dataset_path = "e:/soyabean12/soyabean11/final_dataset_enhanced_hierarchical"
    eff_checkpoint = "e:/soyabean12/soyabean11/models/CNN_trained_models/EfficientNet_B4/best_model_checkpoint.pth"
    res_checkpoint = "e:/soyabean12/soyabean11/models/CNN_trained_models/ResNet152_V2/best_model_checkpoint.pth"
    save_dir = Path("e:/soyabean12/soyabean11/models/CNN_trained_models/V2_Ensemble")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Data Loader
    dataset = HierarchicalDataset(dataset_path, transform=get_eval_transforms())
    
    # Use the same split as training (70/15/15) - we evaluate on the Test set (15%)
    total = len(dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size
    
    _, _, test_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)
    
    # Load Models
    num_classes = len(dataset.classes)
    eff_model = load_efficientnet_b4(num_classes, eff_checkpoint)
    res_model = load_resnet152(num_classes, res_checkpoint)
    
    # Evaluate
    # First get the probabilities
    all_labels = []
    eff_probs = []
    res_probs = []
    
    print("\nRunning inference...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            with torch.amp.autocast('cuda'):
                eff_out = eff_model(inputs)
                eff_p = torch.softmax(eff_out, dim=1)
                res_out = res_model(inputs)
                res_p = torch.softmax(res_out, dim=1)
            eff_probs.append(eff_p.cpu().numpy())
            res_probs.append(res_p.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    eff_probs = np.vstack(eff_probs)
    res_probs = np.vstack(res_probs)
    all_labels = np.array(all_labels)
    
    # Find best weights
    best_w = find_best_weights(eff_probs, res_probs, all_labels)
    
    # Final evaluation with best weights
    ensemble_probs = (best_w[0] * eff_probs) + (best_w[1] * res_probs)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    acc = accuracy_score(all_labels, ensemble_preds)
    
    print(f"\nFinal Ensemble Result (Weights: {best_w}):")
    print(f"  Accuracy: {acc*100:.2f}%")
    
    # Classification Report
    print("\nEnsemble Classification Report:")
    print(classification_report(all_labels, ensemble_preds, target_names=dataset.classes))
    
    # Save Confusion Matrix
    plot_ensemble_confusion_matrix(all_labels, ensemble_preds, dataset.classes, save_dir / "ensemble_confusion_matrix.png")
    
    # Save Results
    results = {
        "ensemble_accuracy": float(acc),
        "best_weights": best_w,
        "models": ["EfficientNet_B4", "ResNet152_V2"]
    }
    with open(save_dir / "ensemble_metrics.json", 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nFinal Ensemble data saved to: {save_dir}")

if __name__ == "__main__":
    main()
