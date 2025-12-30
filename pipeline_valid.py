from dataset import load_text, tokenize, build_vocab, encode

text = load_text("data/wikitext-2/train.txt")
tokens = tokenize(text)
word_to_idx, idx_to_word = build_vocab(tokens)
encoded = encode(tokens, word_to_idx)

print("Vocab size:", len(word_to_idx))
print("First 20 tokens:", tokens[:20])
print("First 20 encoded:", encoded[:20])