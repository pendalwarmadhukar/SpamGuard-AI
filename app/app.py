import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

# Page Configurations (Must be the first Streamlit command)
st.set_page_config(page_title="SpamGuard AI", page_icon="🛡️", layout="centered")

# Download NLTK data required for preprocessing
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

download_nltk_data()

ps = PorterStemmer()

# Text Preprocessing Function (must match exactly what was used in training)
def transform_text(text):
    text = text.lower()
    # Remove HTML tags (for emails)
    text = re.sub('<.*?>', '', text)
    # Tokenization and removing non-alphanumeric
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y[:]
    y.clear()
    
    # Remove stopwords
    for i in text:
        if i not in stopwords.words('english'):
            y.append(i)
            
    text = y[:]
    y.clear()
    
    # Stemming
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)

# UI Header
st.title("🛡️ SpamGuard AI")
st.markdown("**Machine Learning Powered SMS/Email Spam Detection**")
st.divider()

# Load saved Models
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), '..', 'vectorizer.pkl')

try:
    with open(VECTORIZER_PATH, 'rb') as f:
        tfidf = pickle.load(f)
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    models_loaded = True
except FileNotFoundError:
    models_loaded = False
    st.error("Error: Could not find `model.pkl` or `vectorizer.pkl`. Please run the Jupyter Notebook training pipeline first to generate these files.")

# Input Area
input_message = st.text_area("Enter your SMS or Email message below to analyze:", height=200, placeholder="e.g., Congratulations! You have won a $1,000 gift card. Click here to claim your prize.")

if st.button("🔍 Analyze Message"):
    if not models_loaded:
        st.warning("Models are missing. Please train the model first.")
    elif not input_message.strip():
        st.warning("Please enter a message to analyze.")
    else:
        with st.spinner("Analyzing message content..."):
            # 1. Preprocess
            transformed_input = transform_text(input_message)
            
            # 2. Vectorize
            vector_input = tfidf.transform([transformed_input])
            
            # 3. Predict & Get Probability
            result = model.predict(vector_input)[0]
            probability = model.predict_proba(vector_input)[0]
            
            # 4. Display
            st.divider()
            
            spam_prob = probability[1] * 100
            ham_prob = probability[0] * 100

            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Classification")
                if result == 1:
                    st.error("⚠️ SPAM DETECTED")
                else:
                    st.success("✅ LEGITIMATE (HAM)")
                    
            with col2:
                st.subheader("Confidence")
                st.info(f"Spam Probability: **{spam_prob:.2f}%**\nHam Probability: **{ham_prob:.2f}%**")
                
            # Breakdown
            st.markdown("### Analysis Breakdown")
            st.write(f"**Cleaned Tokens:** `{transformed_input}`")
            
st.markdown("---")
st.caption("Powered by Scikit-Learn | Algorithms: TF-IDF, Naive Bayes/Random Forest")
