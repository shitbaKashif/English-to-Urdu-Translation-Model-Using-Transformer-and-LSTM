import os
import re
import torch
from collections import Counter, defaultdict

class BPETokenizer:
    """
    Byte Pair Encoding (BPE) tokenizer for both English and Urdu text
    """
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.special_tokens = {
            "<PAD>": 0,  # Padding token
            "<SOS>": 1,  # Start of sentence token
            "<EOS>": 2,  # End of sentence token
            "<UNK>": 3   # Unknown token
        }
        self.token_to_id = {token: idx for token, idx in self.special_tokens.items()}
        self.id_to_token = {idx: token for token, idx in self.special_tokens.items()}
        self.merges = {}
        self.vocab = {}

    def train(self, texts):
        """
        Train the BPE tokenizer on a list of texts
        """
        # Initialize vocabulary with characters
        word_freqs = Counter()
        for text in texts:
            word_freqs.update(text.split())
        
        # Initialize each word as a sequence of characters
        self.vocab = {token: list(token) for token in word_freqs.keys()}
        
        # Count pairs
        pairs = self._count_pairs(self.vocab, word_freqs)
        
        # Merge pairs until vocab_size is reached or no more pairs
        num_merges = self.vocab_size - len(self.special_tokens)
        for i in range(num_merges):
            if not pairs:
                break
                
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Update vocabulary and merges
            self._merge_pair(best_pair, self.vocab, word_freqs)
            self.merges[best_pair] = len(self.token_to_id)
            
            # Add new token to vocabulary
            new_token = best_pair[0] + best_pair[1]
            self.token_to_id[new_token] = len(self.token_to_id)
            self.id_to_token[self.token_to_id[new_token]] = new_token
            
            # Update pair counts
            pairs = self._count_pairs(self.vocab, word_freqs)
            
            if i % 1000 == 0:
                print(f"Merge {i}/{num_merges}: {best_pair} -> {new_token}")

    def _count_pairs(self, vocab, word_freqs):
        """
        Count the frequency of adjacent pairs in the vocabulary
        """
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            word_tokens = vocab[word]
            for i in range(len(word_tokens) - 1):
                pairs[(word_tokens[i], word_tokens[i+1])] += freq
        return pairs
    
    def _merge_pair(self, pair, vocab, word_freqs):
        """
        Merge a pair of tokens in the vocabulary
        """
        for word in vocab:
            word_tokens = vocab[word]
            new_tokens = []
            i = 0
            while i < len(word_tokens):
                if i < len(word_tokens) - 1 and (word_tokens[i], word_tokens[i+1]) == pair:
                    new_tokens.append(word_tokens[i] + word_tokens[i+1])
                    i += 2
                else:
                    new_tokens.append(word_tokens[i])
                    i += 1
            vocab[word] = new_tokens
    
    def tokenize(self, text):
        """
        Tokenize a text using the learned BPE merges
        """
        words = text.split()
        tokens = []
        
        for word in words:
            # Start with characters
            word_tokens = list(word)
            
            # Apply merges
            while len(word_tokens) > 1:
                pairs = [(word_tokens[i], word_tokens[i+1]) for i in range(len(word_tokens)-1)]
                
                # Find the pair with the highest priority (first occurrence in merges)
                pair_scores = [(pair, self.merges.get(pair, float('inf'))) for pair in pairs]
                best_pair, best_score = min(pair_scores, key=lambda x: x[1])
                
                if best_pair not in self.merges:
                    break
                    
                # Apply the merge
                new_tokens = []
                i = 0
                while i < len(word_tokens):
                    if i < len(word_tokens) - 1 and (word_tokens[i], word_tokens[i+1]) == best_pair:
                        new_tokens.append(word_tokens[i] + word_tokens[i+1])
                        i += 2
                    else:
                        new_tokens.append(word_tokens[i])
                        i += 1
                word_tokens = new_tokens
            
            tokens.extend(word_tokens)
        
        # Add EOS token
        tokens.append("<EOS>")
        
        # Convert tokens to IDs
        ids = []
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                ids.append(self.special_tokens["<UNK>"])
        
        return ids
    
    def detokenize(self, ids):
        """
        Convert token IDs back to text
        """
        tokens = [self.id_to_token.get(id, "<UNK>") for id in ids if id != self.special_tokens["<PAD>"]]
        
        # Remove special tokens
        tokens = [token for token in tokens if token not in ["<SOS>", "<EOS>", "<PAD>", "<UNK>"]]
        
        # Join tokens to form the text
        text = ''.join(tokens)
        
        return text
    
    def save(self, path):
        """
        Save tokenizer to a file
        """
        torch.save({
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token,
            'merges': self.merges,
            'vocab': self.vocab,
            'vocab_size': self.vocab_size
        }, path)
    
    def load(self, path):
        """
        Load tokenizer from a file
        """
        data = torch.load(path)
        self.token_to_id = data['token_to_id']
        self.id_to_token = data['id_to_token']
        self.merges = data['merges']
        self.vocab = data['vocab']
        self.vocab_size = data['vocab_size']
