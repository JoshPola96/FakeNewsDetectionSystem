import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import wordcloud as wc
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string
from IPython.display import display


def download_nltk_resources():
    resources = ['wordnet', 'stopwords', 'punkt', 'omw-1.4', 'punkt_tab']
    for resource in resources:
        try:
            nltk.data.find(f'corpora/{resource}')
            print(f"Resource '{resource}' is already downloaded.")
        except LookupError:
            print(f"Downloading resource '{resource}'...")
            nltk.download(resource)

def generate_wordcloud(text, title):
    wordcloud = wc.WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(" ".join(text))
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=20)
    plt.axis('off')
    plt.show()

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(f"[{string.punctuation}]", " ", text)
    text = text.lower()    
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

def visualize_comparison(original_df, processed_df):
    original_df['text_length'] = original_df['text'].str.len()
    processed_df['text_length'] = processed_df['text'].str.len()

    print("\nOriginal Dataset (First 5 Rows):")
    display(original_df[['text', 'label']].iloc[:5])
    print("\nPreprocessed Dataset (First 5 Rows):")
    display(processed_df[['text', 'label']].iloc[:5])
    
    plt.figure(figsize=(12, 6))
    sns.kdeplot(original_df['text_length'], label='Original Text Length', color='blue', fill=True, alpha=0.3)
    sns.kdeplot(processed_df['text_length'], label='Preprocessed Text Length', color='orange', fill=True, alpha=0.3)
    plt.title('Text Length Distribution: Original vs Preprocessed')
    plt.xlabel('Text Length')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.legend()
    plt.show()

    processed_df.drop(columns=['text_length'], inplace=True)
    return processed_df 


if __name__ == "__main__":

    fake_news_df = pd.read_csv('News_dataset/Fake.csv')
    fake_news_df['label'] = 0
    true_news_df = pd.read_csv('News_dataset/True.csv')
    true_news_df['label'] = 1

    print("True News Dataset:")
    display(true_news_df.head())
    print("Fake News Dataset:")
    display(fake_news_df.head())

    true_news_cleaned = true_news_df.copy()
    fake_news_cleaned = fake_news_df.copy()

    # Dataset before cleaning
    news_df = pd.concat([true_news_cleaned, fake_news_cleaned], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    news_df['text'] = news_df['title'] + " " + news_df['text']
    news_df.dropna(subset=['text'], inplace=True)
    news_df = news_df[news_df['text'].str.strip() != '']

    print("Cleaning up text...")
    true_news_cleaned['text'] = true_news_cleaned['text'].apply(preprocess_text)
    fake_news_cleaned['text'] = fake_news_cleaned['text'].apply(preprocess_text)
    print("Cleaning up title...")
    true_news_cleaned['title'] = true_news_cleaned['title'].apply(preprocess_text)
    fake_news_cleaned['title'] = fake_news_cleaned['title'].apply(preprocess_text)
    
    print("Generating WordClouds...")
    generate_wordcloud(true_news_cleaned['title'], "Word Cloud for True Titles")
    generate_wordcloud(true_news_cleaned['text'], "Word Cloud for True Segments")
    generate_wordcloud(fake_news_cleaned['title'], "Word Cloud for Fake Titles")
    generate_wordcloud(fake_news_cleaned['text'], "Word Cloud for Fake Segments")

    print("Processing dataset...")
    processed_news_df = pd.concat([true_news_cleaned, fake_news_cleaned], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    processed_news_df.drop(columns=['date', 'subject'], errors='ignore', inplace=True)

    processed_news_df['text'] = processed_news_df['title'] + " " + processed_news_df['text']
    processed_news_df.drop(columns=['title'], inplace=True)
    processed_news_df.dropna(subset=['text'], inplace=True)
    processed_news_df = processed_news_df[processed_news_df['text'].str.strip() != '']

    processed_news_df = visualize_comparison(news_df, processed_news_df)    
    
    print("Saving preprocessed data to CSV...")
    processed_news_df.to_csv('processed_news_df.csv', index=False)

