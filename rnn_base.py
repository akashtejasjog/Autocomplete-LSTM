import numpy as np

# -----------------------
# Helpers
# -----------------------
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

def one_hot(idx, size):
    v = np.zeros((size, 1))
    v[idx] = 1
    return v


# -----------------------
# RNN Model
# -----------------------
class SimpleRNN:
    def __init__(self, vocab_size, hidden_size, lr=0.01):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.lr = lr

        self.Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(vocab_size, hidden_size) * 0.01

        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((vocab_size, 1))

    def forward(self, inputs, targets, h_prev):
        xs, hs, ps = {}, {}, {}
        hs[-1] = h_prev
        loss = 0

        for t in range(len(inputs)):
            xs[t] = one_hot(inputs[t], self.vocab_size)

            hs[t] = np.tanh(
                self.Wxh @ xs[t] +
                self.Whh @ hs[t - 1] +
                self.bh
            )

            logits = self.Why @ hs[t] + self.by
            ps[t] = softmax(logits)

            loss += -np.log(ps[t][targets[t], 0])

        return loss, xs, hs, ps

    def backward(self, xs, hs, ps, targets):
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        dh_next = np.zeros_like(hs[0])

        for t in reversed(range(len(xs))):
            dy = ps[t].copy()
            dy[targets[t]] -= 1

            dWhy += dy @ hs[t].T
            dby += dy

            dh = self.Why.T @ dy + dh_next
            dh_raw = (1 - hs[t] ** 2) * dh

            dbh += dh_raw
            dWxh += dh_raw @ xs[t].T
            dWhh += dh_raw @ hs[t - 1].T

            dh_next = self.Whh.T @ dh_raw

        # gradient clipping
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        # update
        self.Wxh -= self.lr * dWxh
        self.Whh -= self.lr * dWhh
        self.Why -= self.lr * dWhy
        self.bh  -= self.lr * dbh
        self.by  -= self.lr * dby


# -----------------------
# Toy data
# -----------------------
text = "hellohello"
chars = list(set(text))
vocab_size = len(chars)

char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for ch, i in char_to_ix.items()}

data = [char_to_ix[ch] for ch in text]

# -----------------------
# Training
# -----------------------
rnn = SimpleRNN(vocab_size, hidden_size=16, lr=0.1)
h_prev = np.zeros((16, 1))

for epoch in range(200):
    inputs = data[:-1]
    targets = data[1:]

    loss, xs, hs, ps = rnn.forward(inputs, targets, h_prev)
    rnn.backward(xs, hs, ps, targets)

    h_prev = hs[len(inputs) - 1]

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, loss: {loss:.4f}")
