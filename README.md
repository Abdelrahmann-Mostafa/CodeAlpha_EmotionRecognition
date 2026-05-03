# 🎙️ Speech Emotion Recognition (SER)

A deep learning project for recognizing human emotions from speech audio, built as part of the **CodeAlpha Machine Learning Internship**. Three architectures are implemented and compared: a vanilla LSTM, a Bidirectional LSTM, and a CNN + BiLSTM hybrid.

---

## 📁 Dataset

- **[RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)** — Ryerson Audio-Visual Database of Emotional Speech and Song
- 8 emotion classes: `Neutral`, `Calm`, `Happy`, `Sad`, `Angry`, `Fearful`, `Disgust`, `Surprised`

---

## 🔧 Feature Extraction

Each audio file is loaded with a 3-second window (0.5s offset) and the following features are extracted using **Librosa** and concatenated into a single feature vector:

| Feature | Description |
|---|---|
| **MFCCs** (40) | Mel-Frequency Cepstral Coefficients — captures timbral texture |
| **Chroma** (12) | Pitch class energy distribution |
| **Mel Spectrogram** (128) | Frequency content over time on the mel scale |

---

## 🧠 Models

### Model 1 — LSTM
- Single-layer LSTM (hidden size: 128)
- Trained with Adam optimizer (lr=0.001)
- Early stopping (patience=30)

### Model 2 — Bidirectional LSTM (BiLSTM)
- 2-layer BiLSTM (hidden size: 128)
- Processes sequence in both directions
- Trained with Adam optimizer (lr=0.01)

### Model 3 — CNN + BiLSTM
- 1D Conv layer → BatchNorm → ReLU → MaxPool → Dropout
- Feeds into 2-layer BiLSTM (hidden size: 128)
- Dropout regularization (0.3) at multiple stages
- Trained with Adam optimizer (lr=0.004)

---

## 📊 Evaluation

All models are evaluated using:
- Classification report (Precision, Recall, F1-Score per class)
- Confusion matrix (heatmap visualization)
- Training/Validation loss and accuracy curves
- Side-by-side model comparison plots

---

## 📈 Results

### Model Comparison

| Model | Train Acc | Val Acc | Train Loss | Val Loss |
|---|---|---|---|---|
| LSTM | 47.61% | 42.01% | 1.5309 | 1.6175 |
| BiLSTM | 83.64% | 71.18% | 0.4913 | 0.8781 |
| **CNN + BiLSTM** | **98.44%** | **84.72%** | **0.1078** | **0.5978** |

### LSTM — Per-Class Report (Val Accuracy: 42%)

| Emotion | Precision | Recall | F1-Score |
|---|---|---|---|
| Neutral | 0.69 | 0.64 | 0.67 |
| Calm | 0.33 | 0.77 | 0.47 |
| Happy | 0.48 | 0.37 | 0.42 |
| Sad | 0.36 | 0.43 | 0.39 |
| Angry | 0.51 | 0.33 | 0.40 |
| Fearful | 0.00 | 0.00 | 0.00 |
| Disgust | 0.38 | 0.18 | 0.24 |
| Surprised | 0.33 | 0.47 | 0.39 |
| **Weighted Avg** | **0.41** | **0.42** | **0.40** |

> The CNN + BiLSTM architecture achieved the best performance, improving validation accuracy from **42% (LSTM) → 71% (BiLSTM) → 85% (CNN+BiLSTM)** by combining local feature extraction via 1D convolutions with sequential modeling via Bidirectional LSTM layers.

---

## 🛠️ Tech Stack

| Category | Libraries |
|---|---|
| Deep Learning | PyTorch |
| Audio Processing | Librosa, Soundfile |
| Data & Preprocessing | NumPy, Pandas, Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Dataset Download | opendatasets, kagglehub |

---

## 🚀 Getting Started

```bash
# Install dependencies
pip install librosa tensorflow soundfile opendatasets torch scikit-learn matplotlib seaborn kagglehub

# Run the notebook
jupyter notebook CodeAlpha_EmotionRec.ipynb
```

> The notebook will automatically download the RAVDESS dataset from Kaggle. You'll need a Kaggle API key (`kaggle.json`) in your working directory.

---

## 📂 Project Structure

```
CodeAlpha_EmotionRecognition/
│
├── CodeAlpha_EmotionRec.ipynb   # Main notebook (EDA, preprocessing, training, evaluation)
└── README.md
```

---

## 🏷️ Topics

`speech-emotion-recognition` `deep-learning` `lstm` `bilstm` `cnn` `pytorch` `librosa` `mfcc` `audio-classification` `ravdess`
