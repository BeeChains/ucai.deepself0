import torch
from torch.optim import Adam
from .model import UCModel
from .utils import get_data_loader, compute_accuracy

def train(model, train_loader, test_loader, epochs=10, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=model.eta)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data.view(-1, 784))
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update recursive parameters
            S, tau, eta = model.update_params(loss.item())
            for param_group in optimizer.param_groups:
                param_group['lr'] = eta
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: Loss={loss.item():.4f}, S={S:.4f}, tau={tau:.4f}, eta={eta:.4f}")
        
        # Evaluate
        model.eval()
        train_acc = compute_accuracy(model, train_loader, device)
        test_acc = compute_accuracy(model, test_loader, device)
        print(f"Epoch {epoch} - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    return model

if __name__ == "__main__":
    train_loader, test_loader = get_data_loader(batch_size=64)
    model = UCModel()
    trained_model = train(model, train_loader, test_loader)