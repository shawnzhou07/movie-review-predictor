import torch
import os
from torch.utils.data import Dataset, DataLoader
from tokenizer import load_tokenizer, build_vocab_to_id, encode


def load_data(data_path, vocab_to_id, merges, max_tokens=None):
    text = ""
    for filename in os.listdir(data_path):
        if filename.endswith(".txt"):
            with open(os.path.join(data_path, filename), "r") as f:
                text += f.read()
    
    tokens = encode(text, vocab_to_id, merges)
    
    if max_tokens:
        tokens = tokens[:max_tokens]
    
    return tokens


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