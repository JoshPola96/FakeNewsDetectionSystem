import streamlit as st
import pandas as pd
import joblib
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# Load model and vectorizer
try:
    model = joblib.load('best_XGBoost.joblib')
    vectorizer = joblib.load('best_tfidf_extractor.joblib')
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")
    st.stop()

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetical characters
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

# Streamlit app
st.title('📰 Fake News Detector')
st.subheader('Analyze news articles or segments to detect if they are real or fake.')

# Input Section
st.markdown("### Enter News Text Below:")
news = st.text_area("Paste the news segment or title you want to analyze.", height=150)

if st.button('Analyze'):
    if not news.strip():
        st.warning("Please enter some text to analyze.")
    else:
        try:
            st.info("Processing your input...")
            preprocessed_news = preprocess_text(news)
            
            # Make prediction
            prediction = model.predict(vectorizer.transform([preprocessed_news]))
            prediction_label = 'FAKE' if prediction[0] == 0 else 'REAL'
            
            # Display result
            st.success(f"Prediction: **{prediction_label}**")
            
            # Confidence score (if available)
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(vectorizer.transform([preprocessed_news]))[0]
                confidence = max(probabilities)
                st.write(f"Confidence: {confidence:.2f}")
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")

# Footer
st.markdown("---")
st.caption("Developed using Python and Streamlit.")
