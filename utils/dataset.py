import torch
from torch.utils.data import Dataset
class TranslationDataset(Dataset):
    def __init__(self, src_path, tgt_path, src_tokenizer, tgt_tokenizer, max_len=100):
        self.src_sentences = open(src_path, encoding='utf-8').read().strip().split('\n')
        self.tgt_sentences = open(tgt_path, encoding='utf-8').read().strip().split('\n')
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        # Use tokenizer's `tokenize` method instead of `encode`
        src = self.src_tokenizer.tokenize(self.src_sentences[idx])
        tgt = self.tgt_tokenizer.tokenize(self.tgt_sentences[idx])
        
        src = src[:self.max_len]
        tgt = tgt[:self.max_len]
        
        return torch.tensor(src), torch.tensor(tgt)
