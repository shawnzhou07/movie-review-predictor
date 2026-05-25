import pickle

# Load vocab
with open('models/tokenizer_vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# Load merges
with open('models/tokenizer_merges.pkl', 'rb') as f:
    merges = pickle.load(f)

print(f"Loaded vocabulary: {len(vocab)} tokens")
print(f"Loaded merges: {len(merges)} rules")

# Show some examples
print("\nSample tokens:")
print(sorted(list(vocab))[:20])  # first 20 alphabetically

print("\nSample merge rules:")
for i, (pair, new_token) in enumerate(merges[:10]):
    print(f"  {i+1}. {pair} -> '{new_token}'")