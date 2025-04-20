# Fake News Detection System

## Overview
This project implements a machine learning pipeline to classify news articles as "Real" or "Fake" using advanced NLP techniques and optimized machine learning models. The system achieves high accuracy through careful preprocessing, feature engineering, and model optimization.

## Key Features
- **Text Preprocessing Pipeline**: Cleans and normalizes text data using NLTK
- **Advanced Feature Extraction**: Utilizes optimized TF-IDF with n-grams
- **Model Optimization**: Bayesian hyperparameter tuning for XGBoost and MLP classifiers
- **Visual Analytics**: Includes word clouds and distribution visualizations
- **Deployment Ready**: Streamlit web interface for real-time predictions

## Technical Skills Demonstrated
### Natural Language Processing
- Text cleaning (HTML removal, punctuation handling)
- Tokenization and lemmatization
- Stopword removal
- N-gram feature extraction
- TF-IDF vectorization optimization

### Machine Learning
- Hyperparameter optimization using Bayesian search
- Neural Networks (MLP) implementation
- Gradient Boosting (XGBoost) implementation
- Model evaluation metrics (precision, recall, F1-score)
- Confusion matrix visualization

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/fake-news-detector.git
```

2. Install dependencies:

3. Download NLTK resources:
```python
import nltk
nltk.download(['wordnet', 'stopwords', 'punkt', 'omw-1.4'])
```

## Usage
1. Preprocess the data:
```python
python preprocessing.py
```

2. Train and evaluate models:
```python
python model_training.py
```

3. Run the web app:
```python
streamlit run app.py
```
