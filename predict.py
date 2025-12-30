import torch
import torch.nn.functional as F

from model import LSTMAutocomplete

# =============================
# Device
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
MODEL_PATH = "checkpoints/lstm_autocomplete.pt"
VOCAB_PATH = "checkpoints/vocab.pt"

# =============================
# Load vocab
# =============================
vocab_data = torch.load(VOCAB_PATH, map_location="cpu")
word_to_idx = vocab_data["word_to_idx"]
idx_to_word = vocab_data["idx_to_word"]

vocab_size = len(word_to_idx)
print("Vocab size:", vocab_size)

# =============================
# Load model
# =============================
model = LSTMAutocomplete(vocab_size=vocab_size)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

print("Model loaded from checkpoint")

# =============================
# Helpers
# =============================
def encode_words(words):
    unk = word_to_idx["<unk>"]
    return [word_to_idx.get(w, unk) for w in words]

def decode_words(indices):
    return [idx_to_word[i] for i in indices]

# =============================
# Predict next word
# =============================
@torch.no_grad()
def predict_next(words, temperature=1.0, top_k=20):
    """
    words: list[str]
    """
    x = torch.tensor(encode_words(words), dtype=torch.long).unsqueeze(0).to(device)

    logits, _ = model(x)
    logits = logits.squeeze(0) / temperature

    if top_k is not None:
        values, indices = torch.topk(logits, top_k)
        probs = torch.zeros_like(logits)
        probs[indices] = F.softmax(values, dim=0)
    else:
        probs = F.softmax(logits, dim=0)

    next_idx = torch.multinomial(probs, 1).item()
    return idx_to_word[next_idx]

# =============================
# Generate text
# =============================
def generate(prompt, max_words=50, temperature=1.0):
    words = prompt.lower().split()

    for _ in range(max_words):
        next_word = predict_next(words, temperature=temperature)
        words.append(next_word)

        if next_word == "<eos>":
            break

    return " ".join(words)

# =============================
# Interactive mode
# =============================
if __name__ == "__main__":
    print("\nType a prompt (or Ctrl+C to quit)\n")

    while True:
        prompt = input(">> ")
        out = generate(prompt, max_words=50, temperature=0.9)
        print(out)
        print()
