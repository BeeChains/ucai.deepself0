# src/evaluate.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add parent dir to path

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from src.model import UCModel  # Absolute import
from src.utils import get_data_loader  # Absolute import

def evaluate_model(model_path="uc_model.pth", device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Load data
    _, test_loader = get_data_loader(batch_size=64)
    
    # Initialize and load model
    model = UCModel(input_dim=784, hidden_dim=512, output_dim=10)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    # Collect predictions and targets
    all_preds = []
    all_targets = []
    all_losses = []
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data.view(-1, 784))
            loss = criterion(output, target)
            
            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_losses.extend(loss.cpu().numpy())
    
    # Metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_targets))
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds))
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    
    # Loss Curve (per batch)
    plt.figure(figsize=(10, 5))
    plt.plot(all_losses, label="Loss per sample")
    plt.title("Test Loss Distribution")
    plt.xlabel("Sample Index")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    evaluate_model()