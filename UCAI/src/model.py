import torch
import torch.nn as nn
import torch.nn.functional as F

class UCModel(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, output_dim=10):
        super(UCModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Neural state (QSM)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_normal_(self.fc1.weight)  # Better initialization
        self.norm1 = nn.LayerNorm(hidden_dim)     # Stabilize training
        
        # Correlation layer (EN) - attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        
        # Relational adapter (PRKP)
        self.relational = nn.Linear(hidden_dim, hidden_dim)
        nn.init.kaiming_normal_(self.relational.weight)
        
        # Output layer
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        nn.init.kaiming_normal_(self.fc2.weight)
        
        # Recursive parameters
        self.S = torch.tensor(1.0)
        self.tau = 0.5
        self.eta = 0.001  # Lower initial learning rate for stability
        
    def forward(self, x):
        # Step 1: Initialization (QSM)
        h = self.norm1(F.relu(self.fc1(x)))  # Add normalization
        
        # Step 2: Correlation Propagation (EN)
        h_att, _ = self.attention(h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0))
        h = h + h_att.squeeze(0)  # Residual connection
        
        # Step 3: Relational Adaptation (PRK)
        h_prime = F.tanh(self.relational(h))
        
        # Step 4: Collapse (QGS simulation)
        h_double_prime = self._collapse(h_prime)
        
        # Step 5: Output
        out = self.fc2(h_double_prime)
        return out
    
    def _collapse(self, h):
        """Improved collapse: Soft thresholding for smoother pruning."""
        threshold = torch.quantile(torch.abs(h), self.tau)
        return F.relu(h - threshold) - F.relu(-h - threshold)  # Symmetric soft threshold
    
    def update_params(self, loss):
        """Feedback Accelerator: Cap growth to prevent instability."""
        alpha, beta, k = 0.05, 0.05, 0.05  # Reduced for stability
        self.S = torch.min(self.S + alpha * self.S**2, torch.tensor(10.0))  # Cap S at 10
        self.tau = max(self.tau / (1 + beta * self.S), 0.1)  # Minimum tau
        self.eta = min(self.eta * (1 + k * self.S), 0.1)    # Cap eta at 0.1
        return self.S, self.tau, self.eta