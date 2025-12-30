import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# --- Import your pipeline ---
from dataset import (
    load_text,
    tokenize,
    build_vocab,
    encode,
    WordSequenceDataset
)

from model import LSTMAutocomplete

# =============================
# Device (Apple Silicon safe)
# =============================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

# =============================
# Paths
# =============================
DATA_PATH = "data/wikitext-2/train.txt"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(CHECKPOINT_DIR, "lstm_autocomplete.pt")
VOCAB_PATH = os.path.join(CHECKPOINT_DIR, "vocab.pt")

# =============================
# Hyperparameters (SAFE START)
# =============================
seq_len = 35
batch_size = 32
embed_dim = 256
hidden_dim = 512
num_layers = 2
num_epochs = 5
learning_rate = 3e-4

# =============================
# Load + preprocess text
# =============================
print("Loading text...")
text = load_text(DATA_PATH)

print("Tokenizing...")
tokens = tokenize(text)

print("Building vocab...")
word_to_idx, idx_to_word = build_vocab(tokens, min_freq=2)

print("Vocab size:", len(word_to_idx))

print("Encoding tokens...")
encoded = encode(tokens, word_to_idx)

# Save vocab so prediction works later
torch.save(
    {
        "word_to_idx": word_to_idx,
        "idx_to_word": idx_to_word,
    },
    VOCAB_PATH
)
print("Saved vocab to:", VOCAB_PATH)

# =============================
# Dataset + DataLoader
# =============================
dataset = WordSequenceDataset(encoded, seq_len)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,   # IMPORTANT for macOS
    pin_memory=False
)

print("Dataset size:", len(dataset))
print("Batches per epoch:", len(dataloader))

# =============================
# Model
# =============================
model = LSTMAutocomplete(
    vocab_size=len(word_to_idx),
    embed_dim=embed_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# =============================
# Training Loop
# =============================
model.train()

for epoch in range(num_epochs):
    total_loss = 0.0

    for batch_idx, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits, _ = model(X)
        loss = criterion(logits, y)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 200 == 0:
            print(
                f"Epoch {epoch+1} | "
                f"Batch {batch_idx}/{len(dataloader)} | "
                f"Loss {loss.item():.4f}"
            )

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} COMPLETE | Avg Loss: {avg_loss:.4f}")

    # Save checkpoint each epoch
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch + 1,
            "loss": avg_loss,
        },
        MODEL_PATH
    )
    print("Saved checkpoint to:", MODEL_PATH)

print("Training finished.")
