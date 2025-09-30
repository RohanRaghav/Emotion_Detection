🎭 Multi-Modal Emotion Recognition (Audio + Text)

This project implements emotion recognition from both audio (speech) and text, using classical machine learning and deep learning approaches.

Audio pipeline: Speech → Feature Extraction (Librosa) → ML Classifiers (MLP, Stacking, Voting, XGBoost)

Text pipeline: Text → Tokenization → Deep Learning Models (BERT, BiLSTM) → Multi-label Emotion Prediction

📂 Project Structure
.
├── Datasets/
│   ├── EMO-DB/wav/             # Raw EMO-DB audio files
│   └── IEMOCAP/
│       ├── IEMOCAP_audio/      # IEMOCAP audio files
│       └── iemocapTrans.csv    # Metadata CSV
├── processed_data/             # Organized audio files by emotion
├── audio_train.py # Audio pipeline (ML)
├── main.py        # Text pipeline (BERT + BiLSTM)
├── bert_local/                 # Local BERT checkpoint
├── results/                    # Output from Trainer
├── audio_model_stacking.pkl     # Trained audio ML model
├── bilstm_emotion_model.pt      # Trained BiLSTM model
├── bert_emotion_model/          # Saved BERT model
├── requirements.txt            # Python dependencies
└── README.md

🚀 Features
Audio Pipeline

Organizes EMO-DB and IEMOCAP datasets into processed_data/ by emotion.

Extracts audio features:

MFCCs + Delta + Delta-Delta

Chroma

Spectral features (Centroid, Bandwidth, Contrast, Rolloff)

RMS Energy & Pitch statistics

Performs data augmentation (pitch shift, time-stretch, noise).

Handles class imbalance with SMOTE.

Trains multiple classifiers:

MLP, Stacking, Voting, XGBoost

Evaluates using confusion matrices & classification reports.

Saves trained model + scaler + label encoder.

Text Pipeline

Supports multi-label emotion classification from text.

Uses BERT fine-tuned for multi-label classification.

Uses BiLSTM with embedding & padding for sequence learning.

Evaluation metrics: Accuracy, F1 (micro/macro), Precision, Recall.

Predict emotions interactively for user-input sentences.

📦 Installation
git clone https://github.com/your-username/multimodal-emotion-recognition.git
cd multimodal-emotion-recognition
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
pip install -r requirements.txt

🛠 Usage
1️⃣ Audio Pipeline
python audio_emotion_recognition.py


Organizes datasets into processed_data/

Extracts features, trains classifiers, saves audio_model_stacking.pkl

Generates confusion matrices: confusion_matrix_MLP.png, confusion_matrix_Stacking.png, etc.

Inference:

import joblib
stack_clf, le, scaler = joblib.load("audio_model_stacking.pkl")
# Extract features from new audio file
# Predict using stack_clf

2️⃣ Text Pipeline
python text_emotion_main.py


Trains BERT & BiLSTM models

Saves bert_emotion_model/ and bilstm_emotion_model.pt

Supports interactive inference:

Enter a sentence (or 'quit'): I am so happy today!
BERT Prediction: ['joy', 'excitement']
BiLSTM Prediction: ['happiness', 'excitement']


Load models for inference:

from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load BERT
tokenizer = BertTokenizer.from_pretrained("bert_emotion_model")
model = BertForSequenceClassification.from_pretrained("bert_emotion_model")
# Run predict_with_bert(text)

⚠️ Notes

Audio pipeline requires EMO-DB or IEMOCAP datasets.

Text pipeline requires test.tsv with columns: text, label, id.

GPU is recommended for BERT/BiLSTM training.

Multi-label text classification uses sigmoid + BCEWithLogitsLoss.

📊 Results

Audio models evaluated using confusion matrices & classification reports.

Text models evaluated with accuracy, F1-micro, F1-macro.

👨‍💻 Author

Rohan Raghav – Full Stack Developer & Machine Learning Enthusiast

Organized hackathons & AI events

Projects: Speech & Text Emotion Recognition, NLP, ML & Deep Learning
