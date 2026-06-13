# 🛡️ VeriText AI — Fake News Detection Platform

> Real-Time NLP Misinformation Detection Powered by Machine Learning

---

## 📌 Overview

**VeriText AI** is an end-to-end fake news detection web application built with Streamlit. It uses a Logistic Regression model trained on TF-IDF features to classify news articles as **Real** or **Fake** in real time, complete with confidence scores, NLP analytics, and keyword explainability.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **ML Classification** | Logistic Regression trained on TF-IDF vectorized text |
| 🔤 **NLP Pipeline** | Lowercasing, stopword removal, lemmatization |
| 📊 **NLP Analytics** | Word count, character count, unique words, reading time |
| 🎯 **Confidence Score** | Probability-based confidence with visual progress bar |
| 🚨 **Risk Levels** | HIGH / MEDIUM / LOW risk flags for fake predictions |
| 🔍 **Keyword Explainability** | Detects sensational language patterns |
| 📜 **Prediction History** | Session-level log of all predictions |
| 📑 **Demo Articles** | 4 sample articles to test instantly |

---

## 🧠 ML Pipeline

```
Raw News Article
      ⬇
Text Cleaning (lowercase → remove special chars → stopwords → lemmatize)
      ⬇
TF-IDF Vectorization
      ⬇
Logistic Regression
      ⬇
Prediction + Confidence Score
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Vinaygoud288/veritext_fake_news_detection.git
cd veritext_fake_news_detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the model file

Place `veritext_model.pkl` in the root directory. The pickle file must contain:

```python
{
    "model": <trained LogisticRegression>,
    "Feature Extraction": <fitted TfidfVectorizer>
}
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
veritext-ai/
├── app.py                  # Main Streamlit application
├── veritext_model.pkl      # Trained model bundle (model + vectorizer)
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🛠️ Tech Stack

- **Frontend / UI** — Streamlit
- **ML Model** — Scikit-learn (Logistic Regression)
- **Feature Extraction** — TF-IDF Vectorizer
- **NLP Preprocessing** — NLTK (stopwords, lemmatization, tokenization)
- **Data & Visualization** — Pandas, NumPy, Matplotlib
- **Model Persistence** — Joblib

---

## 📦 Dependencies

```
pandas==2.2.3
numpy==2.2.3
matplotlib==3.10.0
seaborn==0.13.2
scikit-learn==1.9.0
nltk==3.9.4
streamlit==1.58.0
joblib==1.4.2
```

---

## ⚠️ Risk Level Logic

Risk levels are only assigned to **FAKE** predictions:

| Confidence | Risk Level |
|---|---|
| > 90% | 🔴 HIGH |
| 70–90% | 🟡 MEDIUM |
| < 70% | 🟢 LOW |

Real news predictions always return **LOW** risk.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
  Built for ML Engineer Portfolio &nbsp;•&nbsp; NLP &nbsp;•&nbsp; Machine Learning &nbsp;•&nbsp; TF-IDF &nbsp;•&nbsp; Streamlit
</div>
