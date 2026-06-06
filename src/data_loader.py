import torch
import os
from torch.utils.data import Dataset, DataLoader
from tokenizer import load_tokenizer, build_vocab_to_id, encode
from datasets import load_from_disk


def load_data(data_path, vocab_to_id, merges, max_tokens=None):
    dataset = load_from_disk(data_path)
    stories = dataset["train"]
    
    all_tokens = []
    
    for i, story in enumerate(stories):
        # Tokenize each story individually (much faster!)
        story_tokens = encode(story["text"], vocab_to_id, merges)
        all_tokens.extend(story_tokens)
        
        if i % 500 == 0:
            print(f"  Tokenized {i} stories ({len(all_tokens)} tokens)")
        
        # Stop if we have enough
        if max_tokens and len(all_tokens) >= max_tokens:
            all_tokens = all_tokens[:max_tokens]
            break
    
    return all_tokens


class TextDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.seq_len = seq_len
        self.tokens = torch.tensor(tokens, dtype=torch.long)
    
    def __len__(self):
        return len(self.tokens) // self.seq_len
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end]
        
        input = chunk[:-1]
        target = chunk[1:]
        
        return input, target


def create_dataloader(tokens, seq_len, batch_size, shuffle=True):
    dataset = TextDataset(tokens, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader