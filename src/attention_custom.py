import torch
import torch.nn as nn
import math
import time

class SingleHeadAttention(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        
        # Q, K, V weight matrices
        self.Wq = nn.Linear(embed_dim, head_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, head_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, head_dim, bias=False)
        
        # Scaling factor
        self.scale = 1.0 / math.sqrt(head_dim)
    
    def forward(self, x):
        # Create Q, K, V
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores * self.scale
        
        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply to values
        output = torch.matmul(attn_weights, V)
        
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Create attention heads
        self.heads = nn.ModuleList([
            SingleHeadAttention(embed_dim, self.head_dim)
            for _ in range(num_heads)
        ])
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
    
    def forward(self, x):
        # Run all heads
        head_outputs = [head(x) for head in self.heads]
        
        # Concatenate
        concat = torch.cat(head_outputs, dim=-1)
        
        # Output projection
        output = self.out_proj(concat)
        
        return output


# Test code with timing
if __name__ == "__main__":
    num_runs = 100
    
    # Create custom multi-head attention
    custom_attn = MultiHeadAttention(embed_dim=512, num_heads=8)
    
    # Create fake input
    x = torch.randn(32, 100, 512)  # (batch, seq_len, embed_dim)
    
    # Warm up
    _ = custom_attn(x)
    
    # Time it
    start = time.time()
    for _ in range(num_runs):
        output = custom_attn(x)
    elapsed = time.time() - start
    
    # Results
    print(f"Custom Multi-Head Attention")
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total time ({num_runs} runs): {elapsed:.4f}s")
    print(f"Average per run: {elapsed/num_runs*1000:.2f}ms")
    print("✓ Custom multi-head attention works!")