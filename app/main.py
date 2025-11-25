from fastapi import FastAPI
from pydantic import BaseModel
import torch

from app.model import LSTMClassifier
from app.preprocessing import encode_text, word_to_idx, max_len

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = LSTMClassifier(vocab_size=len(word_to_idx)).to(device)
model.load_state_dict(torch.load("model/lstm_sentiment.pth", map_location=device))
model.eval()

app = FastAPI(title="NLP Sentiment API")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(input: TextInput):
    seq = encode_text(input.text).to(device)

    with torch.no_grad():
        pred = torch.sigmoid(model(seq)).item()

    label = "Positive" if pred > 0.5 else "Negative"

    return {
        "input_text": input.text,
        "prediction": label,
        "score": float(pred)
    }
