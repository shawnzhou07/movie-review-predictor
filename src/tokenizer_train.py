def merge_pair(tokens, pair, new_token):
    """Merge all occurrences of a pair into a new token"""
    new_tokens = []
    i = 0
    
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
            new_tokens.append(new_token)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    
    return new_tokens


def train_bpe(text, vocab_size=1000):
    """Train BPE tokenizer on text corpus"""
    
    # Start with characters
    tokens = list(text)
    
    # Track vocabulary and merges
    vocab = set(tokens)
    merges = []
    
    print(f"Starting vocab size: {len(vocab)}")
    print(f"Target vocab size: {vocab_size}")
    
    # Train until we reach target vocab size
    while len(vocab) < vocab_size:
        # Count all adjacent pairs
        pair_counts = {}
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])
            if pair not in pair_counts:
                pair_counts[pair] = 0
            pair_counts[pair] += 1
        
        # Stop if no pairs left
        if not pair_counts:
            print("No more pairs to merge!")
            break
        
        # Find most frequent pair
        most_frequent_pair = max(pair_counts, key=pair_counts.get)
        
        # Create new token
        new_token = most_frequent_pair[0] + most_frequent_pair[1]
        
        # Merge the pair
        tokens = merge_pair(tokens, most_frequent_pair, new_token)
        
        # Update vocab and merges
        vocab.add(new_token)
        merges.append((most_frequent_pair, new_token))
        
        # Progress update
        if len(vocab) % 100 == 0:
            print(f"Vocab size: {len(vocab)}")
    
    print(f"Final vocab size: {len(vocab)}")
    return vocab, merges


# Train on TinyStories
if __name__ == "__main__":
    from datasets import load_from_disk
    
    print("Loading TinyStories dataset...")
    dataset = load_from_disk("data/pretrain/tinystories")
    
    # Combine first 10,000 stories into one big text string
    # (Using all 2M stories would take too long for first test)
    print("Combining text...")
    texts = dataset['train']['text'][:10000]  # first 10k stories
    combined_text = ' '.join(texts)
    
    print(f"Text length: {len(combined_text):,} characters")
    print(f"Sample: {combined_text[:200]}")
    
    # Train BPE tokenizer
    print("\nTraining BPE tokenizer...")
    vocab, merges = train_bpe(combined_text, vocab_size=5000)
    
    # Save vocabulary and merges
    import pickle
    
    with open('models/tokenizer_vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    
    with open('models/tokenizer_merges.pkl', 'wb') as f:
        pickle.dump(merges, f)
    
    print("\nTokenizer saved to models/")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")