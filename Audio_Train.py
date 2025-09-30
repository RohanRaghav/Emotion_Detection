import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import librosa
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import librosa
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
import random
# ======================================================
# STEP 1: ORGANIZE DATASETS INTO processed_data/
# ======================================================
PROCESSED_DIR = "processed_data"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# --- EMO-DB Mapping ---
emo_db_mapping = {
    "W": "Anger",
    "L": "Boredom",
    "E": "Disgust",
    "A": "Fear",
    "F": "Happiness",
    "T": "Sadness",
    "N": "Neutral"
}

emo_db_path = Path("Datasets/EMO-DB/wav")
if emo_db_path.exists():
    for file in emo_db_path.glob("*.wav"):
        emo_code = file.stem[-2]
        if emo_code in emo_db_mapping:
            emotion = emo_db_mapping[emo_code]
            out_dir = Path(PROCESSED_DIR) / emotion
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(file, out_dir / file.name)
    print("✅ EMO-DB organized")
else:
    print("⚠️ EMO-DB path not found:", emo_db_path.resolve())

# --- IEMOCAP Mapping ---
iemocap_mapping = {
    "ang": "Anger",
    "hap": "Happiness",
    "sad": "Sadness",
    "neu": "Neutral",
    "fea": "Fear",
    "dis": "Disgust",
    "fru": "Frustration",
    "exc": "Excited",
    "sur": "Surprise"
}
# Correct dataset paths
iemocap_path = Path("Datasets/IEMOCAP/IEMOCAP_audio")
csv_path = Path("Datasets/IEMOCAP/iemocapTrans.csv")
print("Looking for EMO-DB at:", emo_db_path.resolve())
print("Looking for IEMOCAP audio at:", iemocap_path.resolve())
print("Looking for IEMOCAP CSV at:", csv_path.resolve())

if csv_path.exists():
    df = pd.read_csv(csv_path)


    file_col = "titre"      # utterance ID
    emo_col = "emotion"     # emotion label

    # 🔹 Build lookup dictionary ONCE
    print("\nIndexing IEMOCAP audio files...")
    all_wavs = list(iemocap_path.rglob("*.wav"))
    file_lookup = {f.stem: f for f in all_wavs}  # stem = filename without extension
    print(f"Indexed {len(file_lookup)} audio files.")

    # 🔹 Now process rows
    for _, row in df.iterrows():
        file_name = str(row[file_col])
        emo_code = str(row[emo_col]).strip().lower()

        if emo_code in iemocap_mapping and file_name in file_lookup:
            emotion = iemocap_mapping[emo_code]
            src_file = file_lookup[file_name]

            out_dir = Path(PROCESSED_DIR) / emotion
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_file, out_dir / (file_name + ".wav"))

    print("✅ IEMOCAP organized")
else:
    print("⚠️ IEMOCAP CSV not found:", csv_path.resolve())

# ======================================================
# STEP 2: CHECK FILE DISTRIBUTION
# ======================================================
print("\nChecking processed_data structure...")
for emotion in os.listdir(PROCESSED_DIR):
    emotion_path = Path(PROCESSED_DIR) / emotion
    if emotion_path.is_dir():
        files = list(emotion_path.glob("*.wav"))
        print(f"{emotion}: {len(files)} files")

def augment_audio(y, sr):
    if random.random() < 0.3:
        y = librosa.effects.pitch_shift(y, sr, n_steps=random.choice([-2, -1, 1, 2]))
    if random.random() < 0.3:
        y = librosa.effects.time_stretch(y, rate=random.uniform(0.8, 1.2))
    if random.random() < 0.3:
        y = y + 0.005 * np.random.randn(len(y))
    return y
# ======================================================
# FEATURE EXTRACTION
# ======================================================
def extract_features(file_path, max_pad_len=300):
    y, sr = librosa.load(file_path, sr=None)
    y = augment_audio(y, sr)
    # MFCCs + deltas
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_delta = librosa.feature.delta(mfccs)
    mfccs_delta2 = librosa.feature.delta(mfccs, order=2)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    # Spectral features
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    # Energy
    rms = librosa.feature.rms(y=y)

    # Pitch
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    pitch_std = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0

    # Stack features
    features = np.vstack([
        mfccs, mfccs_delta, mfccs_delta2,
        chroma, spec_centroid, spec_bandwidth, spec_contrast, spec_rolloff, rms
    ])

    # Pad/truncate
    if features.shape[1] < max_pad_len:
        pad_width = max_pad_len - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
    else:
        features = features[:, :max_pad_len]

    # Flatten + add pitch stats
    flat = features.flatten()
    flat = np.concatenate([flat, [pitch_mean, pitch_std]])
    return flat

# ======================================================
# STEP 3: LOAD DATASET
# ======================================================
def load_dataset(dataset_dir):
    X, y = [], []
    for emotion in os.listdir(dataset_dir):
        emo_dir = Path(dataset_dir) / emotion
        if not emo_dir.is_dir():
            continue
        for file in emo_dir.glob("*.wav"):
            try:
                feat = extract_features(str(file))
                X.append(feat)
                y.append(emotion)
            except:
                continue
    return np.array(X), np.array(y)

X, y = load_dataset(PROCESSED_DIR)
print("Dataset Loaded:", X.shape, len(np.unique(y)), "classes")


# ======================================================
# STEP 4: TRAIN MULTIPLE CLASSIFIERS
# ======================================================
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply PCA
pca = PCA(0.95, random_state=42)  # keep 95% variance
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Encode labels
# Encode labels
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

print("After PCA:", X_train.shape, X_test.shape)

# ======================================================
# STEP 4a: BALANCE DATA WITH SMOTE
# ======================================================
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_train, y_train_enc = sm.fit_resample(X_train, y_train_enc)
print("After SMOTE:", X_train.shape, y_train_enc.shape)

# ======================================================
# Classifier 1: MLP
# ======================================================
mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    alpha=1e-4,
    learning_rate_init=0.001,
    max_iter=1500,
    early_stopping=True,
    n_iter_no_change=50,
    random_state=42,
    verbose=True
)


# ======================================================
# Classifier 2: Stacking
# ======================================================
from sklearn.ensemble import StackingClassifier
mlp_stack = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    alpha=1e-4,
    learning_rate_init=0.001,
    max_iter=1500,
    early_stopping=True,
    n_iter_no_change=50,
    random_state=42,
    verbose=False
)

base_models = [
    ("svm", SVC(kernel="rbf", probability=True, C=10, gamma="scale", class_weight="balanced")),
    ("rf", RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42)),
    ("mlp", mlp_stack)
]

stack_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss"
    ),
    passthrough=True,
    n_jobs=-1
)


# ======================================================
# Classifier 3: Voting
# ======================================================
from sklearn.ensemble import VotingClassifier

vote_clf = VotingClassifier(
    estimators=base_models,
    voting="soft"  # uses predicted probabilities
)

# ======================================================
# Classifier 4: XGBoost
# ======================================================
from xgboost import XGBClassifier
xgb = XGBClassifier(
    n_estimators=2000,     # more boosting rounds
    learning_rate=0.01,    # slower but more precise learning
    max_depth=14,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=1,
    reg_lambda=2,
    reg_alpha=1,
    gamma=0.1,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)


# ======================================================
# STEP 4b: TRAIN ALL MODELS
# ======================================================
print("\nTraining MLP...")
mlp.fit(X_train, y_train_enc)

print("\nTraining Stacking...")
stack_clf.fit(X_train, y_train_enc)

print("\nTraining Voting...")
vote_clf.fit(X_train, y_train_enc)

print("\nTraining XGBoost...")
xgb.fit(X_train, y_train_enc)


from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(stack_clf, X, le.transform(y), cv=skf, scoring="accuracy")
print("Stacking CV Accuracy:", scores.mean())
# ======================================================
# STEP 5: SAVE MODEL AND LABEL ENCODER
# ======================================================
MODEL_PATH = "audio_model_stacking.pkl"
joblib.dump((stack_clf, le, scaler), MODEL_PATH)
print(f"✅ Stacking model, LabelEncoder, and Scaler saved as {MODEL_PATH}")


# ======================================================
# STEP 6: EVALUATION
# ======================================================
models = {
    "MLP": mlp,
    "Stacking": stack_clf,
    "Voting": vote_clf,
    "XGBoost": xgb
}

for name, model in models.items():
    print(f"\n🔹 Evaluating {name}...")

    # Predict with model (outputs encoded labels)
    y_pred_enc = model.predict(X_test)         # encoded predictions
    y_pred_labels = le.inverse_transform(y_pred_enc)  # convert to strings

    # True labels (already strings)
    y_true = y_test  

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_labels, labels=le.classes_)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=le.classes_,
                yticklabels=le.classes_,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{name}.png")
    plt.close()

    # Classification report
    print(f"\nClassification Report for {name}:")
    print(classification_report(y_true, y_pred_labels))
