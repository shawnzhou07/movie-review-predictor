from datasets import load_dataset

# Download TinyStories dataset
dataset = load_dataset("roneneldan/TinyStories")

# Save to disk
dataset.save_to_disk("data/pretrain/tinystories")

print(f"Downloaded {len(dataset['train'])} training examples")
print(f"Sample text: {dataset['train'][0]['text'][:200]}")