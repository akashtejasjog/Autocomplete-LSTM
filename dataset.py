from collections import Counter
import torch
from torch.utils.data import Dataset

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().lower()

def tokenize(text):
    lines = text.split("\n")
    tokens = []

    for line in lines:
        line = line.strip()
        if line:
            tokens.extend(line.split())
            tokens.append("<eos>")

    return tokens

def build_vocab(tokens, min_freq=2):
    counts = Counter(tokens)

    vocab = ["<unk>", "<eos>"]
    for word, freq in counts.items():
        if freq >= min_freq and word not in vocab:
            vocab.append(word)

    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}

    return word_to_idx, idx_to_word

def encode(tokens, word_to_idx):
    unk = word_to_idx["<unk>"]
    return [word_to_idx.get(t, unk) for t in tokens]

class WordSequenceDataset(Dataset):
    def __init__(self, encoded_tokens, seq_len):
        self.data = encoded_tokens
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )
