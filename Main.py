# Main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
# 1. Load your data
dataset = load_dataset(
    "csv",
    data_files="test.tsv",
    delimiter="\t",
    column_names=["text", "label", "id"]
)

id2label = {0:"admiration", 1:"amusement", 2:"anger", 3:"annoyance", 4:"approval",
    5:"caring", 6:"confusion", 7:"curiosity", 8:"desire", 9:"disappointment",
    10:"disapproval", 11:"disgust", 12:"embarrassment", 13:"excitement",
    14:"fear", 15:"gratitude", 16:"grief", 17:"joy", 18:"love",
    19:"nervousness", 20:"optimism", 21:"pride", 22:"realization",
    23:"relief", 24:"remorse", 25:"sadness", 26:"surprise", 27:"neutral"}
label2id = {v: k for k, v in id2label.items()}

# 2. Parse multi-labels and binarize
def parse_labels(label):
    if isinstance(label, int):
        return [label]
    return [int(l.strip()) for l in str(label).split(",") if l.strip().isdigit()]

for split in dataset:
    dataset[split] = dataset[split].map(lambda x: {"labels": parse_labels(x["label"])})

# Fit binarizer on all labels in the dataset
all_labels = [l for example in dataset["train"] for l in parse_labels(example["label"])]
mlb = MultiLabelBinarizer(classes=list(id2label.keys()))
mlb.fit([[l] for l in all_labels])  # fit expects a list of lists

def binarize_labels(example):
    binarized = mlb.transform([example["labels"]])[0].astype("float32").tolist()
    return {"labels": binarized}

for split in dataset:
    dataset[split] = dataset[split].map(binarize_labels)

class MultiLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").to(torch.float)   # ✅ ensure float
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# 3. Tokenize for BERT
tokenizer = BertTokenizer.from_pretrained("./bert_local")
bert_model = BertForSequenceClassification.from_pretrained(
    "./bert_local",
    num_labels=len(id2label),
    problem_type="multi_label_classification",
    id2label=id2label,
    label2id=label2id,
)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
    labels = labels.astype(int)

    acc = (preds == labels).mean()
    f1_micro = f1_score(labels, preds, average="micro")
    f1_macro = f1_score(labels, preds, average="macro")

    return {
        "accuracy": acc,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
    }
def encode_batch(batch):
    encodings = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)
    return {**encodings, "labels": batch["labels"]}

encoded_dataset = dataset.map(encode_batch, batched=True)

# Only now set torch format for tokenized dataset
encoded_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
    output_all_columns=False
)

train_dataset = encoded_dataset["train"]

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    logging_steps=10,
)


trainer = MultiLabelTrainer(
    model=bert_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=encoded_dataset["train"],  # or use validation split if you have one
    compute_metrics=compute_metrics
)


print("Training BERT...")
trainer.train()
bert_model.save_pretrained("./bert_emotion_model")
tokenizer.save_pretrained("./bert_emotion_model")

# 4. BiLSTM model
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        h = torch.cat((h[-2,:,:], h[-1,:,:]), dim=1)
        return self.fc(h)

from torchtext.vocab import build_vocab_from_iterator

def yield_tokens(data_iter):
    for example in data_iter:
        yield example["text"].lower().split()

unk_token = "<unk>"

# Use specials to insert <unk> when building vocab
vocab = build_vocab_from_iterator(yield_tokens(dataset["train"]))
if unk_token not in vocab.itos:
    vocab.itos.append(unk_token)
    vocab.stoi[unk_token] = len(vocab.itos) - 1
unk_index = vocab.stoi[unk_token]

def safe_lookup(token):
    return vocab.stoi[token] if token in vocab.stoi else unk_index

def collate_batch(batch):
    text_list, label_list = [], []
    for example in batch:
        tokens = [safe_lookup(token) for token in example["text"].lower().split()]
        text_list.append(torch.tensor(tokens, dtype=torch.long))
        label_list.append(torch.tensor(example["labels"], dtype=torch.float))
    text_list = pad_sequence(text_list, batch_first=True)
    label_list = torch.stack(label_list)
    return text_list, label_list

train_loader = DataLoader(list(dataset["train"]), batch_size=32, shuffle=True, collate_fn=collate_batch)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bilstm_model = BiLSTMClassifier(len(vocab), embed_dim=100, hidden_dim=128, num_classes=len(id2label)).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(bilstm_model.parameters(), lr=1e-3)

print("Training BiLSTM...")
for epoch in range(2):
    bilstm_model.train()
    total_loss = 0
    print(f"\n=== Starting epoch {epoch+1}/{2} ===")
    
    for batch_idx, (text, labels) in enumerate(train_loader):
        text, labels = text.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = bilstm_model(text)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(f"Epoch {epoch+1} Batch {batch_idx+1}/{len(train_loader)} Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

    # 🔹 Evaluation after each epoch
    bilstm_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for text, labels in train_loader:  # use val_loader if you have one
            text, labels = text.to(device), labels.to(device)
            outputs = bilstm_model(text)
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = (all_preds == all_labels).mean()
    f1_micro = f1_score(all_labels, all_preds, average="micro")
    f1_macro = f1_score(all_labels, all_preds, average="macro")

    print(f"Epoch {epoch+1} Accuracy: {acc:.4f}")
    print(f"Epoch {epoch+1} F1-micro: {f1_micro:.4f}")
    print(f"Epoch {epoch+1} F1-macro: {f1_macro:.4f}")
torch.save(bilstm_model.state_dict(), "bilstm_emotion_model.pt")

# 5. Inference
def predict_with_bert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
    outputs = bert_model(**inputs)
    probs = torch.sigmoid(outputs.logits).squeeze()
    pred_indices = (probs > 0.5).nonzero(as_tuple=True)[0].tolist()
    if not pred_indices:
        pred_indices = [torch.argmax(probs).item()]
    return [id2label[i] for i in pred_indices]

def predict_with_bilstm(text):
    tokens = [safe_lookup(token) for token in text.lower().split()]
    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        output = bilstm_model(input_tensor)
        probs = torch.sigmoid(output).squeeze()
        pred_indices = (probs > 0.5).nonzero(as_tuple=True)[0].tolist()
        if not pred_indices:
            pred_indices = [torch.argmax(probs).item()]
    return [id2label[i] for i in pred_indices]

print("\n--- Test Your Models ---")
while True:
    user_input = input("Enter a sentence (or 'quit'): ")
    if user_input.lower() == "quit":
        break
    print("BERT Prediction:", predict_with_bert(user_input))
    print("BiLSTM Prediction:", predict_with_bilstm(user_input))

import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", device)
