import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from PIL import Image, ImageOps
import numpy as np
import os

# ✅ Safe TensorFlow import (important for cloud)
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except:
    TF_AVAILABLE = False

# -------------------------
# Stopwords
# -------------------------
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# -------------------------
# Text Cleaning
# -------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# -------------------------
# TEXT MODEL
# -------------------------
@st.cache_resource
def load_text_model():
    spam = pd.read_csv("spam.csv", encoding='latin-1')
    spam = spam[['v1', 'v2']]
    spam.columns = ['label', 'text']
    spam['label'] = spam['label'].map({'ham': 1, 'spam': 0})

    reviews = pd.read_csv("fake reviews dataset.csv")
    reviews = reviews.iloc[:, [0, -1]]
    reviews.columns = ['text', 'label']

    reviews['label'] = reviews['label'].astype(str).str.lower()
    reviews['label'] = reviews['label'].map({
        'cg': 1, 'or': 0,
        'fake': 0, 'real': 1,
        '1': 1, '0': 0
    })
    reviews = reviews.dropna()

    data = pd.concat([spam, reviews], ignore_index=True)
    data['text'] = data['text'].apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data['text'])
    y = data['label']

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model, vectorizer

# -------------------------
# IMAGE MODEL
# -------------------------
@st.cache_resource
def load_image_model():
    if not TF_AVAILABLE:
        return None

    if os.path.exists("ai_detector_v2.h5"):
        try:
            return tf.keras.models.load_model("ai_detector_v2.h5")
        except:
            return None
    return None

def predict_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    img_array = np.asarray(image).astype(np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_reshaped = img_array[np.newaxis, ...]

    prediction = image_model.predict(img_reshaped)
    score = prediction[0][0]

    if score > 0.5:
        return "AI Generated / Modified", score * 100
    else:
        return "Likely Authentic", (1 - score) * 100

# -------------------------
# INIT
# -------------------------
text_model, vectorizer = load_text_model()
image_model = load_image_model()

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="AI Content Detection System", page_icon="🕵️")

st.sidebar.title("⚙️ Control Panel")
mode = st.sidebar.selectbox("Choose Module", ["Text Analysis", "Image Analysis"])

# =========================
# TEXT MODULE
# =========================
if mode == "Text Analysis":
    st.title("🕵️ AI Text & Spam Detector")

    user_input = st.text_area("✍️ Enter text:")

    if st.button("🔍 Analyze"):
        if user_input:
            text_cleaned = clean_text(user_input)
            vector = vectorizer.transform([text_cleaned])

            prob = text_model.predict_proba(vector)[0]
            fake_score = prob[0]
            real_score = prob[1]

            scam_words = ["win", "lottery", "urgent", "prize", "money"]
            found = [w for w in text_cleaned.split() if w in scam_words]

            if len(found) > 0:
                st.error("🚨 High Risk / Scam Content")
                conf = fake_score
            elif fake_score > 0.9:
                st.error("🚨 Fake Content")
                conf = fake_score
            else:
                st.success("✅ Real Content")
                conf = real_score

            st.write(f"Confidence: {round(conf*100,2)}%")
            st.progress(int(conf * 100))

        else:
            st.warning("Enter some text")

# =========================
# IMAGE MODULE
# =========================
else:
    st.title("🖼️ AI Image Detector")

    if not TF_AVAILABLE:
        st.error("⚠️ TensorFlow not available on cloud")
    else:
        uploaded = st.file_uploader("Upload Image", type=["jpg", "png"])

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image)

            if st.button("Analyze Image"):
                if image_model:
                    label, conf = predict_image(image)

                    if "AI" in label:
                        st.error(label)
                    else:
                        st.success(label)

                    st.write(f"Confidence: {round(conf,2)}%")
                else:
                    st.error("Model file missing")