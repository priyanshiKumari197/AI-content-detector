import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# -------------------------
# 1. System Dependencies & NLTK Setup
# -------------------------
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('stopwords')
        return set(stopwords.words('english'))
    except:
        return set()

stop_words = download_nltk_data()

# -------------------------
# 2. Model Loading (Cached for Performance)
# -------------------------
@st.cache_resource
def load_ai_engine():
    model_path = 'ai_detector_v2.h5'
    if os.path.exists(model_path):
        try:
            # Using tensorflow-cpu internally on Streamlit
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            st.error(f"Engine Error: {e}")
            return None
    return None

# Global Image Model Instance
image_model = load_ai_engine()

# -------------------------
# 3. Text Analysis Logic
# -------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

@st.cache_resource
def load_text_model():
    try:
        # Loading Datasets
        spam = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
        spam.columns = ['label', 'text']
        spam['label'] = spam['label'].map({'ham': 1, 'spam': 0})

        reviews = pd.read_csv("fake reviews dataset.csv")
        # Column Identification
        text_col = next((c for c in reviews.columns if 'text' in c.lower() or 'review' in c.lower()), reviews.columns[0])
        label_col = next((c for c in reviews.columns if 'label' in c.lower() or 'class' in c.lower()), reviews.columns[-1])

        reviews = reviews[[text_col, label_col]]
        reviews.columns = ['text', 'label']
        reviews['label'] = reviews['label'].astype(str).str.lower().str.strip()
        reviews['label'] = reviews['label'].map({'cg': 1, 'or': 0, 'fake': 0, 'real': 1, 'positive': 1, 'negative': 0, '1': 1, '0': 0})
        reviews = reviews.dropna()

        data = pd.concat([spam, reviews], ignore_index=True)
        data['text'] = data['text'].apply(clean_text)

        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(data['text'])
        y = data['label']

        model = LogisticRegression(class_weight='balanced', max_iter=1000)
        model.fit(X, y)
        return model, vectorizer
    except Exception as e:
        st.error(f"Dataset Loading Error: {e}")
        return None, None

text_model, vectorizer = load_text_model()

# -------------------------
# 4. Forensic Core Functions
# -------------------------
def predict_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image).astype(np.float32)
    # Important: Normalization
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_reshaped = img_array[np.newaxis, ...]
    
    prediction = image_model.predict(img_reshaped)
    score = prediction[0][0]
    
    if score > 0.5: 
        return "AI Generated / Modified", score * 100
    else:
        return "Likely Authentic", (1 - score) * 100

# -------------------------
# 5. User Interface (UI)
# -------------------------
st.set_page_config(page_title="Multi-Layered AI Detection", page_icon="🕵️", layout="centered")

st.sidebar.title("⚙️ Control Panel")
app_mode = st.sidebar.selectbox("Choose Analysis Module", ["Text Analysis", "Image Analysis"])

if app_mode == "Text Analysis":
    st.title("🕵️ AI Text & Spam Detector")
    st.info("System Layer: NLP Calibration (90% Threshold Active)")

    content_type = st.selectbox("Select Category", ["Message", "Review"])
    user_input = st.text_area(f"Enter {content_type} Content:")

    if st.button("🔍 Run NLP Scan"):
        if user_input and text_model:
            text_cleaned = clean_text(user_input)
            vector = vectorizer.transform([text_cleaned])
            
            probability = text_model.predict_proba(vector)[0]
            fake_score, real_score = probability[0], probability[1]
            
            strict_scam_words = ["win", "winner", "lottery", "urgent", "cash", "prize", "claim"]
            found_scam = [w for w in text_cleaned.split() if w in strict_scam_words]
            
            # FINAL CALIBRATED LOGIC
            if len(found_scam) > 0:
                st.error(f"🚨 Detection Result: High Risk / Scam Content")
                st.warning(f"Suspicious words flagged: {', '.join(found_scam)}")
                conf = fake_score
            elif fake_score > 0.90:
                st.error(f"🚨 Detection Result: Fake/Spam Content")
                conf = fake_score
            else:
                st.success(f"✅ Detection Result: Authentic Content")
                conf = real_score

            st.write(f"📊 Analysis Confidence: {round(conf*100,2)}%")
            st.progress(int(conf * 100))
        else:
            st.warning("Please enter text to analyze.")

else:
    st.title("🖼️ AI Image Forensic Layer")
    st.info("System Layer: Computer Vision Artifact Analysis.")
    
    uploaded_file = st.file_uploader("Upload Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Analysis Target", use_container_width=True)
        
        if st.button("🔍 Start Forensic Scan"):
            if image_model:
                with st.spinner("Analyzing pixels..."):
                    label, conf = predict_image(image)
                    if "AI" in label:
                        st.error(f"🚨 Result: {label}")
                    else:
                        st.success(f"✅ Result: {label}")
                    st.metric("Confidence", f"{round(conf, 2)}%")
            else:
                st.error("Engine Offline: Model file 'ai_detector_v2.h5' not loaded.")