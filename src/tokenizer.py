import pickle

# ==================== LOADING ====================

def load_tokenizer():
    """Load trained tokenizer from disk"""
    with open('models/tokenizer_vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    
    with open('models/tokenizer_merges.pkl', 'rb') as f:
        merges = pickle.load(f)
    
    return vocab, merges


# ==================== VOCAB MAPPINGS ====================

def build_vocab_to_id(vocab):
    """Create mapping from tokens to IDs"""
    vocab_to_id = {}
    for i, token in enumerate(sorted(vocab)):
        vocab_to_id[token] = i
    return vocab_to_id


def build_id_to_vocab(vocab_to_id):
    """Create reverse mapping from IDs to tokens"""
    id_to_vocab = {}
    for token, id in vocab_to_id.items():
        id_to_vocab[id] = token
    return id_to_vocab


# ==================== MERGE OPERATIONS ====================

def merge_pair(tokens, pair, new_token):
    """Merge all occurrences of a pair (used in encoding)"""
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


def apply_merges(text, merges):
    """Apply BPE merge rules to tokenize text"""
    # Start with characters
    tokens = list(text)
    
    # Apply each merge rule in order
    for pair, new_token in merges:
        tokens = merge_pair(tokens, pair, new_token)
    
    return tokens


# ==================== ENCODE / DECODE ====================

def encode(text, vocab_to_id, merges):
    """Convert text to token IDs"""
    # Apply merges to get tokens
    tokens = apply_merges(text, merges)
    
    # Convert tokens to IDs
    token_ids = []
    for token in tokens:
        if token in vocab_to_id:
            token_ids.append(vocab_to_id[token])
        else:
            # Unknown token (shouldn't happen if tokenizer trained properly)
            token_ids.append(0)  # Use ID 0 as fallback
    
    return token_ids


def decode(token_ids, id_to_vocab):
    """Convert token IDs back to text"""
    # Convert IDs to tokens
    tokens = []
    for token_id in token_ids:
        tokens.append(id_to_vocab[token_id])
    
    # Join tokens into text
    text = ''.join(tokens)
    return text


# ==================== TESTING ====================

if __name__ == "__main__":
    # Load tokenizer
    print("Loading tokenizer...")
    vocab, merges = load_tokenizer()
    vocab_to_id = build_vocab_to_id(vocab)
    id_to_vocab = build_id_to_vocab(vocab_to_id)
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merge rules: {len(merges)}")
    
    # Test encoding
    test_text = "The cat sat on the mat"
    print(f"\nOriginal text: {test_text}")
    
    token_ids = encode(test_text, vocab_to_id, merges)
    print(f"Token IDs: {token_ids}")
    print(f"Number of tokens: {len(token_ids)}")
    
    # Test decoding
    decoded_text = decode(token_ids, id_to_vocab)
    print(f"Decoded text: {decoded_text}")
    
    # Verify round-trip
    print("\n" + "="*50)
    if test_text == decoded_text:
        print("✓ Round-trip successful!")
    else:
        print("✗ Round-trip failed!")
        print(f"  Expected: '{test_text}'")
        print(f"  Got: '{decoded_text}'")