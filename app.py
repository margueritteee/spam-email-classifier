import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd

# Load external CSS
def load_css():
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
    except:
        pass

download_nltk_data()

# Load model and vectorizer
@st.cache_resource
def load_model():
    try:
        model = joblib.load('spam_classifier_model')
        vectorizer = joblib.load('tfidf_vectorizer')
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d{10}|\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

# Page config
st.set_page_config(
    page_title="Spam Detector", 
    page_icon="🛡️", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load CSS
load_css()

# Title with icon
st.markdown("<h1 style='text-align: center;'>🛡️ SPAM DETECTOR</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #b0b0b0; font-size: 1.1rem;'>AI-Powered Email Security System</p>", unsafe_allow_html=True)
st.markdown("---")

# Session state for examples
if 'current_message' not in st.session_state:
    st.session_state.current_message = ""

# Example buttons
st.markdown("<h3 style='color: #00ff88;'>⚡ Quick Test Examples</h3>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    if st.button("✅ SAFE MESSAGE", use_container_width=True):
        st.session_state.current_message = "Hey! Are we still on for dinner tonight? Let me know what time works for you."
with col2:
    if st.button("⚠️ SPAM MESSAGE", use_container_width=True):
        st.session_state.current_message = "CONGRATULATIONS! You've won a $1000 gift card! Click here NOW to claim your prize: http://fake-scam.com/claim"

st.markdown("<br>", unsafe_allow_html=True)

# Main input
st.markdown("<h3 style='color: #00ff88;'>📧 Enter Message to Analyze</h3>", unsafe_allow_html=True)
message_text = st.text_area(
    "",
    height=140,
    placeholder="Paste your email or message here for analysis...",
    label_visibility="collapsed",
    value=st.session_state.current_message
)

st.markdown("---")

# Analyze button
if st.button("🔍 SCAN MESSAGE", type="primary", use_container_width=True):
    if message_text and len(message_text.strip()) > 0:
        model, vectorizer = load_model()
        
        if model and vectorizer:
            with st.spinner("🔄 Analyzing threat level..."):
                # Preprocess
                cleaned = preprocess_text(message_text)
                
                if len(cleaned.strip()) == 0:
                    st.warning("⚠️ MESSAGE EMPTY - No meaningful content detected.")
                else:
                    # Vectorize and predict
                    message_vector = vectorizer.transform([cleaned]).toarray()
                    prediction = model.predict(message_vector)[0]
                    probabilities = model.predict_proba(message_vector)[0]
                    
                    st.markdown("---")
                    st.markdown("<h2 style='text-align: center; color: #00ff88;'>📊 SCAN RESULTS</h2>", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Show result with Gmail-style alerts
                    if prediction == 1:
                        st.error("### 🚨 SPAM DETECTED - THREAT IDENTIFIED")
                        confidence = probabilities[1]
                        st.warning("⚠️ **WARNING:** This message exhibits characteristics of spam/phishing. Exercise extreme caution. Do not click any links or provide personal information.")
                        threat_level = "HIGH RISK"
                    else:
                        st.success("### ✅ SAFE MESSAGE - NO THREATS DETECTED")
                        confidence = probabilities[0]
                        st.info("✓ **VERIFIED:** This message appears to be legitimate. No spam indicators found.")
                        threat_level = "SECURE"
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Metrics in cyber-style cards
                    col_m1, col_m2, col_m3 = st.columns(3)
                    
                    with col_m1:
                        st.metric("🎯 CONFIDENCE", f"{confidence*100:.1f}%")
                    
                    with col_m2:
                        st.metric("🔐 STATUS", threat_level)
                    
                    with col_m3:
                        st.metric("📝 CLASS", "SPAM" if prediction == 1 else "HAM")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Progress bar
                    st.markdown(f"<p style='color: #b0b0b0;'>Threat Confidence Level:</p>", unsafe_allow_html=True)
                    st.progress(float(confidence))
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Show probabilities
                    st.markdown("<h3 style='color: #00ff88;'>📈 Probability Analysis</h3>", unsafe_allow_html=True)
                    prob_df = pd.DataFrame({
                        'Classification': ['SAFE (HAM)', 'SPAM'],
                        'Probability': [probabilities[0], probabilities[1]]
                    })
                    st.bar_chart(prob_df.set_index('Classification'))
                    
                    # Processed text
                    with st.expander("🔍 VIEW PROCESSED DATA"):
                        st.code(cleaned, language=None)
                        st.caption("This is the preprocessed text analyzed by the AI model")
    else:
        st.warning("⚠️ Please enter a message to scan!")

# Sidebar - Cyber theme
with st.sidebar:
    st.markdown("<h2 style='color: #00ff88;'>🛡️ SYSTEM INFO</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("<h3 style='color: #14ffec;'>🤖 AI Model</h3>", unsafe_allow_html=True)
    st.write("**Algorithm:** Support Vector Machine")
    st.write("**Vectorization:** TF-IDF")
    st.write("**Features:** 3,000 words")
    st.write("**Accuracy:** 98.5%")
    st.write("**Precision:** 100%")
    
    st.markdown("---")
    
    st.markdown("<h3 style='color: #ff4444;'>⚠️ SPAM INDICATORS</h3>", unsafe_allow_html=True)
    st.write("🔴 Urgent calls to action")
    st.write("🔴 Suspicious URLs/links")
    st.write("🔴 Prize/money promises")
    st.write("🔴 Poor grammar/spelling")
    st.write("🔴 ALL CAPS text")
    st.write("🔴 Unknown senders")
    st.write("🔴 Request for personal info")
    
    st.markdown("---")
    
    st.markdown("<h3 style='color: #00ff88;'>✅ SAFETY TIPS</h3>", unsafe_allow_html=True)
    st.write("🟢 Verify sender identity")
    st.write("🟢 Check URL destinations")
    st.write("🟢 Never share passwords")
    st.write("🟢 Report suspicious emails")
    st.write("🟢 Use 2FA authentication")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #666; font-size: 0.9rem;'>🔒 Developed by Kezrane Margueritte |Spam Email Classifier</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #444; font-size: 0.8rem;'>SVM-based Detection System</p>", unsafe_allow_html=True)
