import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.layer_norm(x)
        return x


# Test code
if __name__ == "__main__":
    embed_dim = 512
    hidden_dim = 2048
    
    # Test FeedForward
    ff = FeedForward(embed_dim, hidden_dim)
    x = torch.randn(2, 10, embed_dim)
    output = ff(x)
    
    print(f"FeedForward:")
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ FeedForward works!")