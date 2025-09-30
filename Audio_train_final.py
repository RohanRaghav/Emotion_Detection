import os
from pathlib import Path
import numpy as np
import torch
import librosa
from tqdm import tqdm
import collections

from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier

# ======================================================
# LOAD WAV2VEC2 BASE MODEL
# ======================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE)
model.eval()


# ======================================================
# FEATURE EXTRACTION
# ======================================================
def extract_wav2vec_features(file_path):
    try:
        # load wav, resample to 16kHz
        speech, sr = librosa.load(file_path, sr=16000)

        # processor expects batch of inputs
        inputs = processor(speech, sampling_rate=16000,
                           return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model(inputs.input_values.to(DEVICE)).last_hidden_state
            # mean pooling across time dimension
            feat = outputs.mean(dim=1).cpu().numpy().squeeze()

        return feat
    except Exception as e:
        print(f"Feature extraction error in {file_path}: {e}")
        return None


# ======================================================
# DATASET LOADER
# ======================================================
def load_dataset(dataset_dir):
    X, y = [], []
    emotions = os.listdir(dataset_dir)

    for emotion in emotions:
        emo_dir = Path(dataset_dir) / emotion
        if not emo_dir.is_dir():
            continue

        files = list(emo_dir.glob("*.wav"))
        for file in tqdm(files, desc=f"Processing {emotion}", unit="file"):
            feat = extract_wav2vec_features(str(file))
            if feat is not None:
                X.append(feat)
                y.append(emotion)

    return np.array(X), np.array(y)


# ======================================================
# MAIN PIPELINE
# ======================================================
if __name__ == "__main__":
    dataset_path = "processed_data"  # change if needed

    print("Loading dataset...")
    X, y = load_dataset(dataset_path)

    print("Feature shape:", X.shape)
    print("Classes:", set(y))
    print("Class counts:", collections.Counter(y))

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ==================================================
    # CLASSIFIER (MLP)
    # ==================================================
    clf = MLPClassifier(
        hidden_layer_sizes=(512, 256),
        activation="relu",
        solver="adam",
        batch_size=32,
        max_iter=20,
        verbose=True
    )

    print("\nTraining classifier...")
    clf.fit(X_train, y_train)

    # ==================================================
    # EVALUATION
    # ==================================================
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\nTest Accuracy:", round(acc * 100, 2), "%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
