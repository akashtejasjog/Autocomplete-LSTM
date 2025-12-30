import torch
import torch.nn.functional as F
from model import LSTMAutocomplete

# -----------------------
# Configuration
# -----------------------
CHECKPOINT_PATH = "checkpoints/lstm_autocomplete.pt"
VOCAB_PATH = "checkpoints/vocab.pt"

SEQ_LEN = 20
TEMPERATURE = 0.8
TOP_K = 40
REPETITION_PENALTY = 1.2
MAX_GENERATE = 50

# -----------------------
# Device
# -----------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# -----------------------
# Load vocab
# -----------------------
vocab_data = torch.load(VOCAB_PATH)
word_to_idx = vocab_data["word_to_idx"]
idx_to_word = vocab_data["idx_to_word"]

vocab_size = len(word_to_idx)

# -----------------------
# Load model
# -----------------------
model = LSTMAutocomplete(vocab_size=vocab_size)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

print("Model loaded from checkpoint")

# -----------------------
# Sampling helpers
# -----------------------
def apply_repetition_penalty(logits, recent_tokens, penalty):
    for t in set(recent_tokens):
        logits[0, t] /= penalty
    return logits


def top_k_filter(logits, k):
    if k <= 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_val = values[:, -1].unsqueeze(1)
    return torch.where(logits < min_val, torch.full_like(logits, -1e10), logits)


def sample_next(logits, recent_tokens):
    logits = logits / TEMPERATURE
    logits = apply_repetition_penalty(logits, recent_tokens, REPETITION_PENALTY)
    logits = top_k_filter(logits, TOP_K)

    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, 1).item()
    return next_token


# -----------------------
# Generation loop
# -----------------------
def generate(prompt):
    tokens = prompt.lower().split()
    encoded = [word_to_idx.get(w, word_to_idx["<unk>"]) for w in tokens]

    generated = encoded[:]

    with torch.no_grad():
        hidden = None

        for _ in range(MAX_GENERATE):
            context = generated[-SEQ_LEN:]
            x = torch.tensor(context, dtype=torch.long).unsqueeze(0).to(device)

            logits, hidden = model(x, hidden)
            next_token = sample_next(logits, generated[-10:])

            generated.append(next_token)

            if idx_to_word[next_token] == "<eos>":
                break

    words = [idx_to_word[i] for i in generated]
    return " ".join(words)


# -----------------------
# Interactive prompt
# -----------------------
print("\nType a prompt (or Ctrl+C to quit)\n")

while True:
    try:
        prompt = input(">> ").strip()
        if not prompt:
            continue
        print(generate(prompt))
        print()
    except KeyboardInterrupt:
        print("\nBye 👋")
        break
