# 🎭 Multi-Modal Emotion Recognition (Audio + Text)

This project implements **emotion recognition** from both **audio (speech)** and **text**, using classical machine learning and deep learning approaches.

---

## 🔊 Audio Pipeline
**Workflow:** Speech → Feature Extraction → ML Classifiers

- **Feature Extraction:** Using [Librosa](https://librosa.org/)
  - MFCCs + Delta + Delta-Delta
  - Chroma
  - Spectral features (Centroid, Bandwidth, Contrast, Rolloff)
  - RMS Energy & Pitch statistics
- **Data Processing:**
  - Organizes EMO-DB and IEMOCAP datasets into `processed_data/` by emotion
  - Data augmentation: pitch shift, time-stretch, noise
  - Handles class imbalance with SMOTE
- **Classifiers Trained:**
  - MLP, Stacking, Voting, XGBoost
- **Evaluation:**
  - Confusion matrices
  - Classification reports
- **Outputs:**
  - Trained model, scaler, label encoder saved as `audio_model_stacking.pkl`
  - Confusion matrices: `confusion_matrix_MLP.png`, `confusion_matrix_Stacking.png`, etc.

---

## 📝 Text Pipeline
**Workflow:** Text → Tokenization → Deep Learning Models → Multi-label Emotion Prediction

- Supports **multi-label emotion classification**
- Models used:
  - BERT (fine-tuned)
  - BiLSTM with embeddings & padding
- **Evaluation Metrics:** Accuracy, F1 (micro/macro), Precision, Recall
- Supports **interactive inference**:
  ```text
  Enter a sentence (or 'quit'): I am so happy today!
  BERT Prediction: ['joy', 'excitement']
  BiLSTM Prediction: ['happiness', 'excitement']

Models saved as:

-bert_emotion_model/

-bilstm_emotion_model.pt

📂 Project Structure
.
├── Datasets/
│   ├── EMO-DB/wav/           # Raw EMO-DB audio files
│   └── IEMOCAP/
│       ├── IEMOCAP_audio/    # IEMOCAP audio files
│       └── iemocapTrans.csv  # Metadata CSV
├── processed_data/           # Organized audio files by emotion
├── audio_train.py            # Audio pipeline (ML)
├── main.py                   # Text pipeline (BERT + BiLSTM)
├── bert_local/               # Local BERT checkpoint
├── results/                  # Output from Trainer
├── audio_model_stacking.pkl  # Trained audio ML model
├── bilstm_emotion_model.pt   # Trained BiLSTM model
├── bert_emotion_model/       # Saved BERT model
├── requirements.txt          # Python dependencies
└── README.md

🚀 Installation
git clone [https://github.com/your-username/multimodal-emotion-recognition.git](https://github.com/RohanRaghav/Emotion_Detection.git)
cd multimodal-emotion-recognition
python -m venv .venv
# Activate environment
# Linux / Mac
source .venv/bin/activate
# Windows
.venv\Scripts\activate
pip install -r requirements.txt

🛠 Usage
1️⃣ Audio Pipeline
python audio_emotion_recognition.py
Organizes datasets into processed_data/

Extracts features, trains classifiers

Saves audio_model_stacking.pkl

Generates confusion matrices

Inference Example:
import joblib

stack_clf, le, scaler = joblib.load("audio_model_stacking.pkl")
# Extract features from new audio file
# Predict using stack_clf
2️⃣ Text Pipeline
python text_emotion_main.py

Trains BERT & BiLSTM models

Saves bert_emotion_model/ and bilstm_emotion_model.pt
Inference Example:
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

Audio Models: Evaluated with confusion matrices & classification reports

Text Models: Evaluated with accuracy, F1-micro, F1-macro

👨‍💻 Author

Rohan Raghav – Full Stack Developer & Machine Learning Enthusiast

Organized hackathons & AI events

Projects: Speech & Text Emotion Recognition, NLP, ML & Deep Learning