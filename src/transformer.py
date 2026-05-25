import torch
import torch.nn as nn

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
    block = TransformerBlock(embed_dim=512, num_heads=8)
    x = torch.randn(2, 10, 512)
    output = block(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Transformer block works!")