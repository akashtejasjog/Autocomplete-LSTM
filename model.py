import torch
import torch.nn as nn

class LSTMAutocomplete(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)            # (B, T, E)
        out, hidden = self.lstm(emb, hidden)
        last = out[:, -1, :]               # (B, H)
        logits = self.fc(last)             # (B, V)
        return logits, hidden
