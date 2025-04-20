import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import xgboost as xgb
import joblib
import logging
import matplotlib.pyplot as plt
import time
from fake_news_detector_preprocess import preprocess_text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use the best TF-IDF parameters directly
def get_best_tfidf():
    logging.info("Using the best TF-IDF parameters...")
    best_tfidf = TfidfVectorizer(
        max_features=150000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.9,
        dtype=np.float32
    )
    return best_tfidf

# Evaluate the model
def evaluate_model(model, X_test, y_test, model_name):
    logging.info(f"Evaluating {model_name}...")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    logging.info(f"Evaluation complete for {model_name}.")
    return {
        'Model': model_name,
        'Accuracy': report['accuracy'],
        'Precision': report['weighted avg']['precision'],
        'Recall': report['weighted avg']['recall'],
        'F1-Score': report['weighted avg']['f1-score'],
    }, y_pred

# Plot confusion matrix
def plot_confusion_matrix(y_test, y_pred, model_name):
    logging.info(f"Plotting confusion matrix for {model_name}...")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


best_mlp_params = {
    'hidden_layer_sizes': (300, 150),  # Based on best validation scores in logs
    'activation': 'relu',              # Consistent good performance in logs
    'alpha': 1e-5,                     # Best performing regularization value
    'learning_rate_init': 0.001,       # Good convergence rate from logs
    'max_iter': 50,                    # Most models converged within this range
    'early_stopping': True,            # Shown effective in logs
    'validation_fraction': 0.1,        # Standard split that worked well
    'tol': 0.0001,                    # Default tolerance worked well
    'n_iter_no_change': 10            # Good balance from the logs
}

best_xgb_params = {
    'learning_rate': 0.1,           # Moderate learning rate for stability
    'max_depth': 6,                 # Moderate depth to capture complex text patterns
    'n_estimators': 300,            # Good balance of performance vs training time
    'subsample': 0.8,               # Prevent overfitting on text data
    'colsample_bytree': 0.8,        # Help manage high feature dimensionality
    'gamma': 1,                     # Moderate pruning for text classification
    'reg_alpha': 1e-4,             # L1 regularization for sparse text features
    'reg_lambda': 1e-3,            # L2 regularization for stability
}

# Train and evaluate models using pre-defined best parameters
def train_and_evaluate_best_params(X_train, X_test, y_train, y_test):
    results = []
    models_predictions = {}

    # Train XGBoost with best parameters
    logging.info("Training XGBoost with the best parameters...")
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        **best_xgb_params
    )
    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=10,
        verbose=2  # Log every 10 rounds
    )
    xgb_result, xgb_pred = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    results.append(xgb_result)
    models_predictions['XGBoost'] = xgb_model

    # Train MLP with best parameters
    logging.info("Training MLP with the best parameters...")
    mlp_model = MLPClassifier(
        random_state=42,
        verbose=True,
        **best_mlp_params
    )
    mlp_model.fit(X_train, y_train)
    mlp_result, mlp_pred = evaluate_model(mlp_model, X_test, y_test, "MLP")
    results.append(mlp_result)
    models_predictions['MLP'] = mlp_model

    # Plot Confusion Matrices
    for model_name, model in models_predictions.items():
        y_pred = model.predict(X_test)
        plot_confusion_matrix(y_test, y_pred, model_name)

    return pd.DataFrame(results), models_predictions

# Evaluate the best model with a custom dataset
def evaluate_custom_data(custom_data, model, preprocessor):
    logging.info("Evaluating custom data...")
    custom_data_processed = custom_data.apply(preprocess_text)
    custom_data_features = preprocessor.transform(custom_data_processed)
    predictions = model.predict(custom_data_features)
    return predictions

if __name__ == "__main__":
    logging.info("Starting the news classification pipeline...")
    start_time = time.time()

    # Load dataset
    news_df = pd.read_csv('processed_news_df.csv')
    X = news_df['text']
    y = news_df['label']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Use best TF-IDF directly
    logging.info("Initializing TF-IDF vectorizer with best parameters...")
    best_tfidf = get_best_tfidf()
    X_train_features = best_tfidf.fit_transform(X_train)
    X_test_features = best_tfidf.transform(X_test)

    # Save preprocessor
    joblib.dump(best_tfidf, 'best_tfidf_extractor.joblib')

    # Train and evaluate models
    results_df, models_predictions = train_and_evaluate_best_params(X_train_features, X_test_features, y_train, y_test)
    results_df.sort_values(['Accuracy', 'Precision', 'Recall', 'F1-Score'], ascending=False, inplace=True)

    # Save results
    results_df.to_csv('models_evaluation_results.csv', index=False)

    # Save the best model
    best_model_name = results_df.iloc[0]['Model']  # Select the model with the highest accuracy
    best_model = models_predictions[best_model_name]
    joblib.dump(best_model, f'best_{best_model_name}.joblib')

    logging.info(f"Pipeline completed in {(time.time() - start_time) / 60:.2f} minutes")

    # Test on a custom dataset
    custom_news = pd.Series([
        "Donald Trump is the best president in the history of the United States.",
        "COVID-19 vaccines are dangerous and can alter your DNA.",
        "A new study finds that eating chocolate every day is good for your health.",
        "The Earth is flat and NASA is hiding the truth from us.",
        "The moon landing was a hoax.",
        "The COVID-19 pandemic was planned by the government."
    ])
    custom_predictions = evaluate_custom_data(custom_news, best_model, best_tfidf)
    print("Predictions on custom dataset:", custom_predictions)