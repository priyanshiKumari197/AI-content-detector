import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# -------------------------
# 1. System Dependencies
# -------------------------
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# -------------------------
# 2. Text Analysis Logic
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
        spam = pd.read_csv("spam.csv", encoding='latin-1')
        spam = spam[['v1', 'v2']]
        spam.columns = ['label', 'text']
        spam['label'] = spam['label'].map({'ham': 1, 'spam': 0})

        reviews = pd.read_csv("fake reviews dataset.csv")
        text_col = reviews.columns[0]
        label_col = reviews.columns[-1]
        for col in reviews.columns:
            if 'text' in col.lower() or 'review' in col.lower(): text_col = col
            if 'label' in col.lower() or 'class' in col.lower(): label_col = col

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
        st.error(f"Error loading Text Datasets: {e}")
        return None, None

# -------------------------
# 3. Image Forensic Logic
# -------------------------
@st.cache_resource
def load_image_model():
    model_path = 'ai_detector_v2.h5' 
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            return model
        except Exception as e:
            st.error(f"Error loading Image Model: {e}")
            return None
    else:
        st.error("Model file 'ai_detector_v2.h5' not found.")
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
        label = "AI Generated / Modified"
        confidence = score * 100
    else:
        label = "Likely Authentic"
        confidence = (1 - score) * 100
    return label, confidence

# -------------------------
# 4. Initialization
# -------------------------
text_model, vectorizer = load_text_model()
image_model = load_image_model()

# -------------------------
# 5. User Interface (UI)
# -------------------------
st.set_page_config(page_title="AI Content Detection System", page_icon="🕵️", layout="centered")

st.sidebar.title("⚙️ Control Panel")
app_mode = st.sidebar.selectbox("Choose Analysis Module", ["Text Analysis", "Image Analysis"])

# --- MODULE 1: TEXT ANALYSIS (FINAL CALIBRATED LOGIC) ---
if app_mode == "Text Analysis":
    st.title("🕵️ AI Text & Spam Detector")
    st.info("System Layer 1: NLP for Message and Review Authentication.")

    content_type = st.selectbox("Select Content Category", ["Message", "Review"])
    user_input = st.text_area(f"✍️ Enter {content_type} Content:")

    if st.button("🔍 Run NLP Scan"):
        if user_input and text_model:
            text_cleaned = clean_text(user_input)
            vector = vectorizer.transform([text_cleaned])
            
            # Prediction probablity nikalo
            probability = text_model.predict_proba(vector)[0]
            # probability[0] = Fake hone ka chance, probability[1] = Real hone ka chance
            fake_score = probability[0]
            real_score = probability[1]
            
            # High-risk Scam Keywords
            strict_scam_words = ["win", "winner", "lottery", "urgent", "cash", "prize", "claim"]
            found_scam = [w for w in text_cleaned.split() if w in strict_scam_words]
            
            # NEW CALIBRATED LOGIC
            # 1. Agar koi khatarnak scam word mil jaye toh turant BLOCK
            if len(found_scam) > 0:
                st.error(f"🚨 Detection Result: High Risk / Scam Content")
                conf = fake_score
            # 2. Agar model 90% se zyada sure hai ki ye Fake hai, tabhi Error dikhao
            elif fake_score > 0.90:
                st.error(f"🚨 Detection Result: Fake/Spam Content")
                conf = fake_score
            # 3. Baaki sab ke liye 'Authentic' ya 'Likely Authentic'
            else:
                st.success(f"✅ Detection Result: Authentic Content")
                conf = real_score

            st.write(f"📊 Analysis Confidence: {round(conf*100,2)}%")
            st.progress(int(conf * 100))
        else:
            st.warning("Please provide input text for analysis.")
# --- MODULE 2: IMAGE ANALYSIS ---
else:
    st.title("🖼️ AI Image Forensic Layer")
    st.info("System Layer 2: Computer Vision Analysis for Pixel Artifact Detection.")
    
    uploaded_file = st.file_uploader("Upload Target Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded File", use_container_width=True) # UI Fix
        
        if st.button("🔍 Start Forensic Scan"):
            if image_model:
                with st.spinner("Analyzing pixel artifacts..."):
                    label, conf = predict_image(image)
                    if "AI" in label:
                        st.error(f"🚨 Analysis Result: {label}")
                    else:
                        st.success(f"✅ Analysis Result: {label}")
                    st.metric("Detection Confidence", f"{round(conf, 2)}%")
            else:
                st.error("Engine Error: Model file missing.")