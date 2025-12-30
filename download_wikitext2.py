from datasets import load_dataset
import os

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

os.makedirs("data/wikitext-2", exist_ok=True)

splits = {
    "train": dataset["train"],
    "valid": dataset["validation"],
    "test": dataset["test"],
}

for name, split in splits.items():
    path = f"data/wikitext-2/{name}.txt"
    with open(path, "w", encoding="utf-8") as f:
        for line in split["text"]:
            f.write(line + "\n")

print("WikiText-2 downloaded successfully.")