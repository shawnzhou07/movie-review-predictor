import torch
import torch.nn as nn
import math
import time


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


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, embed_dim))
    
    def forward(self, x):
        return self.weight[x]


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, embed_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(max_seq_len, embed_dim))
    
    def forward(self, x):
        seq_len = x.shape[1]
        return x + self.weight[:seq_len]


class SingleHeadAttention(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.Wq = Linear(embed_dim, head_dim)
        self.Wk = Linear(embed_dim, head_dim)
        self.Wv = Linear(embed_dim, head_dim)
        self.scale = 1.0 / math.sqrt(head_dim)
    
    def forward(self, x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, V)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList([
            SingleHeadAttention(embed_dim, self.head_dim)
            for _ in range(num_heads)
        ])
        self.out_proj = Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]
        concat = torch.cat(head_outputs, dim=-1)
        return self.out_proj(concat)


class LayerNorm(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + 1e-8) + self.beta


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        hidden_dim = embed_dim * 4
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.layer_norm1 = LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, hidden_dim)
        self.layer_norm2 = LayerNorm(embed_dim)
    
    def forward(self, x):
        attn_output = self.attention(x)
        x = x + attn_output
        x = self.layer_norm1(x)
        ff_output = self.ff(x)
        x = x + ff_output
        x = self.layer_norm2(x)
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_blocks, max_seq_len):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(max_seq_len, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(num_blocks)
        ])
        self.output = Linear(embed_dim, vocab_size)
    
    def forward(self, x):
        x = self.token_embedding(x)
        x = self.pos_encoding(x)
        for block in self.blocks:
            x = block(x)
        x = self.output(x)
        return x


if __name__ == "__main__":
    num_runs = 10

    model = Transformer(
        vocab_size=5000,
        embed_dim=512,
        num_heads=8,
        num_blocks=12,
        max_seq_len=1000
    )

    token_ids = torch.randint(0, 5000, (2, 10))

    # Warm up
    _ = model(token_ids)

    # Time it
    start = time.time()
    for _ in range(num_runs):
        output = model(token_ids)
    elapsed = time.time() - start

    print(f"Custom Transformer")
    print(f"Input shape:  {token_ids.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total time ({num_runs} runs): {elapsed:.4f}s")
    print(f"Average per run: {elapsed/num_runs*1000:.2f}ms")
    print("✓ Custom Transformer works!")