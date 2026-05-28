import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        return x @ self.weight.T + self.bias


class ReLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.clamp(x, min=0)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        
        self.fc1 = Linear(embed_dim, hidden_dim)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_dim, embed_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Test code
if __name__ == "__main__":
    embed_dim = 512
    hidden_dim = 2048

    ff = FeedForward(embed_dim, hidden_dim)
    x = torch.randn(2, 10, embed_dim)
    output = ff(x)

    print(f"FeedForward (custom):")
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Custom FeedForward works!")