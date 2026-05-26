import torch
import torch.nn as nn
import time

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
    
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return attn_output


# Test code with timing
if __name__ == "__main__":
    num_runs = 100
    
    # Create PyTorch multi-head attention
    pytorch_attn = MultiHeadAttention(embed_dim=512, num_heads=8)
    
    # Create fake input
    x = torch.randn(32, 100, 512)  # (batch, seq_len, embed_dim)
    
    # Warm up
    _ = pytorch_attn(x)
    
    # Time it
    start = time.time()
    for _ in range(num_runs):
        output = pytorch_attn(x)
    elapsed = time.time() - start
    
    # Results
    print(f"PyTorch Multi-Head Attention")
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total time ({num_runs} runs): {elapsed:.4f}s")
    print(f"Average per run: {elapsed/num_runs*1000:.2f}ms")
    print("✓ PyTorch multi-head attention works!")