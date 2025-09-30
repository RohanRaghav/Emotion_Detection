import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pickle
import torch.nn as nn

id2label = {
    0: "admiration", 1: "amusement", 2: "anger", 3: "annoyance", 4: "approval",
    5: "caring", 6: "confusion", 7: "curiosity", 8: "desire", 9: "disappointment",
    10: "disapproval", 11: "disgust", 12: "embarrassment", 13: "excitement",
    14: "fear", 15: "gratitude", 16: "grief", 17: "joy", 18: "love",
    19: "nervousness", 20: "optimism", 21: "pride", 22: "realization",
    23: "relief", 24: "remorse", 25: "sadness", 26: "surprise", 27: "neutral"
}

# -------------------------------
# Load BERT model and tokenizer
# -------------------------------
tokenizer = BertTokenizer.from_pretrained("./bert_emotion_model")
bert_model = BertForSequenceClassification.from_pretrained("./bert_emotion_model")
bert_model.eval()

# -------------------------------
# Load BiLSTM vocab
# -------------------------------
try:
    with open("bilstm_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
except FileNotFoundError:
    vocab = None
    print("⚠️ Warning: bilstm_vocab.pkl not found, BiLSTM predictions disabled.")

# -------------------------------
# Define BiLSTM model
# -------------------------------
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        h = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
        return self.fc(h)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Load BiLSTM model only if vocab available
# -------------------------------
if vocab:
    unk_token = "<unk>"
    unk_index = vocab.stoi.get(unk_token, 0)

    bilstm_model = BiLSTMClassifier(
        len(vocab.itos), embed_dim=100, hidden_dim=128, num_classes=len(id2label)
    ).to(device)

    bilstm_model.load_state_dict(torch.load("bilstm_emotion_model.pt", map_location=device))
    bilstm_model.eval()
else:
    bilstm_model = None

# -------------------------------
# Utility for token lookup
# -------------------------------
def safe_lookup(token):
    return vocab.stoi[token] if token in vocab.stoi else unk_index

# -------------------------------
# Prediction functions
# -------------------------------
def predict_with_bert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze()
        pred_indices = (probs > 0.5).nonzero(as_tuple=True)[0].tolist()
        if not pred_indices:
            pred_indices = [torch.argmax(probs).item()]
    return [id2label[i] for i in pred_indices]

def predict_with_bilstm(text):
    if not bilstm_model:
        return ["BiLSTM model not available"]
    tokens = [safe_lookup(token) for token in text.lower().split()]
    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        output = bilstm_model(input_tensor)
        probs = torch.sigmoid(output).squeeze()
        pred_indices = (probs > 0.5).nonzero(as_tuple=True)[0].tolist()
        if not pred_indices:
            pred_indices = [torch.argmax(probs).item()]
    return [id2label[i] for i in pred_indices]

# -------------------------------
# Interactive loop
# -------------------------------
print("\n--- Real-time Emotion Prediction ---")
while True:
    user_input = input("Enter a sentence (or 'quit'): ")
    if user_input.lower() == "quit":
        break
    print("BERT Prediction:", predict_with_bert(user_input))
    print("BiLSTM Prediction:", predict_with_bilstm(user_input))
