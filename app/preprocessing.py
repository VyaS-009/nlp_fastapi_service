import re
import torch
import json

config = json.load(open("app/vocab.json"))
word_to_idx = config["word_to_idx"]
max_len = config["max_len"]

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text.strip()

def encode_text(text):
    tokens = clean_text(text).split()
    seq = [word_to_idx.get(t,1) for t in tokens]
    if len(seq) < max_len:
        seq += [0] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return torch.tensor([seq], dtype=torch.long)
