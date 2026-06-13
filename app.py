import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
import joblib
import re
import nltk

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="VeriText AI",
    page_icon="🛡️",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #050816,
        #0f172a,
        #1e1b4b
    );
}

.block-container {
    padding-top: 2rem;
}

.hero-title{
    font-size:60px;
    font-weight:800;
    text-align:center;
    color:#00d4ff;
}

.hero-sub{
    font-size:20px;
    text-align:center;
    color:#cbd5e1;
}

.hero-desc{
    text-align:center;
    color:#94a3b8;
    margin-bottom:30px;
}

.metric-card{
    background:#0f172a;
    padding:20px;
    border-radius:18px;
    border:1px solid rgba(0,212,255,.25);
    text-align:center;
}

.metric-value{
    font-size:28px;
    color:#00d4ff;
    font-weight:bold;
}

.metric-label{
    color:white;
}

.result-card{
    padding:25px;
    border-radius:18px;
    text-align:center;
}

.real-card{
    background:#052e16;
    border:2px solid #22c55e;
}

.fake-card{
    background:#450a0a;
    border:2px solid #ef4444;
}

.info-card{
    background:#0f172a;
    padding:20px;
    border-radius:18px;
    border:1px solid rgba(0,212,255,.15);
}

.footer{
    text-align:center;
    color:#94a3b8;
    padding:20px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# NLTK DOWNLOADS
# =====================================================

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# =====================================================
# TEXT PREPROCESSING
# =====================================================

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def text_preprocessing(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]+", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_assets():
    bundle = joblib.load("veritext_model.pkl")
    model = bundle["model"]
    vectorizer = bundle["Feature Extraction"]
    return model, vectorizer

try:
    model, vectorizer = load_assets()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.error(f"Model loading error: {e}")

# =====================================================
# SAMPLE ARTICLES
# =====================================================

samples = {

"Sample 1":
"""
The government announced a new infrastructure investment program to improve transportation systems and create employment opportunities nationwide.
""",

"Sample 2":
"""
BREAKING NEWS! Scientists secretly discovered a miracle fruit capable of curing every disease overnight. Governments are hiding the truth.
""",

"Sample 3":
"""
The central bank released its quarterly report indicating stable inflation and moderate economic growth throughout the fiscal year.
""",

"Sample 4":
"""
SHOCKING! Celebrity reveals a secret method to become rich within 24 hours. Experts claim banks don't want people to know this trick.
"""
}

# =====================================================
# SESSION STATE
# =====================================================

if "article_text" not in st.session_state:
    st.session_state.article_text = ""

if "history" not in st.session_state:
    st.session_state.history = []

# =====================================================
# HERO SECTION
# =====================================================

st.markdown("""
<div class="hero-title">
🛡️ VeriText AI
</div>

<div class="hero-sub">
Fake News Intelligence Platform
</div>

<div class="hero-desc">
Real-Time NLP Misinformation Detection Powered by Machine Learning
</div>
""", unsafe_allow_html=True)

# =====================================================
# KPI CARDS
# =====================================================

c1,c2,c3,c4 = st.columns(4)

cards = [
    ("ML","Model"),
    ("TF-IDF","Features"),
    ("NLP","Pipeline"),
    ("AI","Prediction")
]

for col,(value,label) in zip([c1,c2,c3,c4],cards):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")

# =====================================================
# SAMPLE BUTTONS
# =====================================================

st.subheader("📑 Demo Articles")

b1,b2,b3,b4 = st.columns(4)

for name,col in zip(samples.keys(), [b1,b2,b3,b4]):
    with col:
        if st.button(name, use_container_width=True):
            st.session_state.article_text = samples[name]

# =====================================================
# MAIN SECTION
# =====================================================

left,right = st.columns([2,1])

with left:

    article = st.text_area(
        "📰 Enter News Article",
        key="article_text",
        height=300
    )

    analyze = st.button(
        "🚀 Analyze Article",
        use_container_width=True
    )

with right:

    st.markdown("""
    <div class="info-card">

    <h3>Model Information</h3>

    ✅ Logistic Regression

    ✅ TF-IDF Vectorization

    ✅ NLP Classification

    ✅ Streamlit Deployment

    </div>
    """, unsafe_allow_html=True)

# =====================================================
# PREDICTION
# =====================================================

if analyze:

    if not article.strip():
        st.warning("Please enter article text.")
        st.stop()

    if not MODEL_LOADED:
        st.stop()

    # FIX 1: Preprocess text before vectorizing (same as training pipeline)
    cleaned_article = text_preprocessing(article)
    X = vectorizer.transform([cleaned_article])

    prediction = model.predict(X)[0]

    try:
        confidence = float(
            np.max(
                model.predict_proba(X)[0]
            )
        )
    except:
        confidence = 0.90

    label = "REAL NEWS" if prediction == 1 else "FAKE NEWS"

    # FIX 2: Risk level only applies to FAKE predictions
    if prediction == 0:
        if confidence > 0.90:
            risk = "HIGH"
        elif confidence > 0.70:
            risk = "MEDIUM"
        else:
            risk = "LOW"
    else:
        risk = "LOW"

    st.subheader("Prediction Result")

    if prediction == 1:

        st.markdown(f"""
        <div class="result-card real-card">
        <h1>✅ {label}</h1>
        <h3>Confidence: {confidence:.2%}</h3>
        <h3>Risk Level: {risk}</h3>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="result-card fake-card">
        <h1>🚨 {label}</h1>
        <h3>Confidence: {confidence:.2%}</h3>
        <h3>Risk Level: {risk}</h3>
        </div>
        """, unsafe_allow_html=True)

    # =================================================
    # CONFIDENCE
    # =================================================

    st.subheader("🎯 Confidence Score")

    st.progress(int(confidence * 100))

    st.metric(
        "Prediction Confidence",
        f"{confidence:.2%}"
    )

    # =================================================
    # ANALYTICS
    # =================================================

    st.subheader("📊 NLP Analytics")

    words = article.split()

    a1,a2,a3,a4 = st.columns(4)

    a1.metric("Words", len(words))
    a2.metric("Characters", len(article))
    a3.metric("Unique Words", len(set(words)))
    a4.metric(
        "Read Time",
        f"{max(1,len(words)//200)} min"
    )

    # =================================================
    # TOP KEYWORDS
    # =================================================

    st.subheader("🔍 Top Keywords")

    clean_words = re.sub(
        r'[^a-zA-Z ]',
        ' ',
        article.lower()
    ).split()

    common = Counter(
        clean_words
    ).most_common(10)

    if common:

        labels = [x[0] for x in common]
        values = [x[1] for x in common]

        fig, ax = plt.subplots(figsize=(8,4))

        ax.barh(labels, values)

        ax.set_title("Most Frequent Terms")

        st.pyplot(fig)

    # =================================================
    # EXPLANATION
    # =================================================

    st.subheader("🤖 AI Explanation")

    suspicious = [
        "breaking",
        "shocking",
        "secret",
        "exclusive",
        "miracle",
        "banned"
    ]

    found = []

    for word in suspicious:
        if word in article.lower():
            found.append(word)

    if found:

        st.warning(
            "Potential sensational language detected."
        )

        for item in found:
            st.write(
                f"⚠ Keyword detected: {item}"
            )

    else:

        st.success(
            "No major sensational indicators found."
        )

    # =================================================
    # HISTORY
    # =================================================

    st.session_state.history.insert(
        0,
        {
            "Time":
            datetime.now().strftime("%H:%M:%S"),

            "Prediction":
            label,

            "Confidence":
            round(confidence * 100, 2)
        }
    )

# =====================================================
# PIPELINE
# =====================================================

st.divider()

st.subheader("⚙️ Machine Learning Pipeline")

st.markdown("""
**Raw News Article**

⬇️

**Text Cleaning**

⬇️

**TF-IDF Vectorization**

⬇️

**Logistic Regression**

⬇️

**Prediction**

⬇️

**Confidence Score**
""")

# =====================================================
# HISTORY TABLE
# =====================================================

st.divider()

st.subheader("📜 Prediction History")

if len(st.session_state.history):

    st.dataframe(
        pd.DataFrame(
            st.session_state.history
        ),
        use_container_width=True
    )

else:

    st.info("No predictions yet.")

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.title("VeriText AI")

    if MODEL_LOADED:                                       
        st.success("🟢 Model Status :\n Loaded Successfully")
    else:
        st.error("🟢 Model Status :\n Not Loaded")

    st.markdown("---")

    st.write("""
### Features

✅ Fake News Detection

✅ TF-IDF Features

✅ NLP Analytics

✅ Confidence Score

✅ Prediction History

✅ Explainability

✅ Streamlit Deployment
""")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown("""
<div class="footer">

<h4>Built for ML Engineer Portfolio</h4>

NLP • Machine Learning • TF-IDF • Streamlit

</div>
""", unsafe_allow_html=True)