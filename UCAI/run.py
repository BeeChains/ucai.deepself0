import torch
from src.model import UCModel
from src.train import train
from src.utils import get_data_loader

if __name__ == "__main__":
    # Load data
    train_loader, test_loader = get_data_loader(batch_size=64)
    
    # Initialize model
    model = UCModel(input_dim=784, hidden_dim=512, output_dim=10)
    
    # Train with recursive self-improvement
    trained_model = train(model, train_loader, test_loader, epochs=10)
    
    # Save the model
    torch.save(trained_model.state_dict(), "uc_model.pth")
    print("Model training complete and saved as 'uc_model.pth'")