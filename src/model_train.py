import torch
import torch.nn as nn
import time
import os
from tokenizer import Tokenizer
from transformer_pytorch import Transformer
from data_loader import load_data, create_dataloader
import pickle


def train():
    # ============ PARAMETERS ============
    vocab_size    = 16000
    embed_dim     = 512
    num_heads     = 8
    num_blocks    = 12
    max_seq_len   = 256
    batch_size    = 32
    learning_rate = 3e-4
    num_epochs    = 3
    
    # ============ LOAD TOKENIZER ============
    tokenizer = Tokenizer()
    tokenizer.load("models/tokenizer_vocab.pkl", "models/tokenizer_merges.pkl")
    print("✓ Tokenizer loaded")
    
    # ============ LOAD DATA ============
    tokens = load_data(
        data_path="data/pretrain/tinystories/",
        tokenizer=tokenizer,
        max_tokens=1000000  # start with 1M tokens for testing
    )
    print(f"✓ Loaded {len(tokens)} tokens")
    
    # ============ CREATE DATALOADER ============
    dataloader = create_dataloader(tokens, max_seq_len, batch_size)
    print(f"✓ Created {len(dataloader)} batches")
    
    # ============ CREATE MODEL ============
    model = Transformer(vocab_size, embed_dim, num_heads, num_blocks, max_seq_len)
    print(f"✓ Model created")
    
    # ============ LOSS + OPTIMIZER ============
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # ============ TRAINING LOOP ============
    for epoch in range(num_epochs):
        epoch_start = time.time()
        total_loss = 0
        num_batches = 0
        
        for input_batch, target_batch in dataloader:
            # Forward pass
            output = model(input_batch)
            
            # Reshape for loss calculation
            # output: (batch, seq_len, vocab_size) → (batch * seq_len, vocab_size)
            # target: (batch, seq_len) → (batch * seq_len)
            output = output.view(-1, vocab_size)
            target_batch = target_batch.view(-1)
            
            # Calculate loss
            loss = loss_fn(output, target_batch)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track progress
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches % 100 == 0:
                print(f"  Epoch {epoch+1} | Batch {num_batches}/{len(dataloader)} | Loss: {loss.item():.4f}")
        
        # Epoch summary
        avg_loss = total_loss / num_batches
        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch+1} complete | Avg Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")
        
        # Save checkpoint
        torch.save(model.state_dict(), f"models/transformer_epoch{epoch+1}.pt")
        print(f"✓ Checkpoint saved")
    
    print("✓ Training complete!")


if __name__ == "__main__":
    train()