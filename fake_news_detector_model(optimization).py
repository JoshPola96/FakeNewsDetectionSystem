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

def convert_hidden_layers(layer_string):
    """Convert string representation of hidden layers to tuple of integers"""
    try:
        # Remove any spaces and split by comma
        layers = [int(x.strip()) for x in layer_string.split(',')]
        return tuple(layers)
    except Exception as e:
        logging.error(f"Error converting hidden layers: {str(e)}")
        return (100,)  # Default fallback value

# Model Optimization using Bayesian Optimization
def optimize_model(X_train, y_train, model, search_spaces, n_iter=50):
    logging.info("Starting Bayesian Optimization...")
    
    # Create a wrapper class to handle the string conversion
    class MLPWrapper(MLPClassifier):
        def set_params(self, **params):
            if 'hidden_layer_sizes' in params and isinstance(params['hidden_layer_sizes'], str):
                params['hidden_layer_sizes'] = convert_hidden_layers(params['hidden_layer_sizes'])
            return super().set_params(**params)
    
    # Replace the original model with wrapped version
    if isinstance(model, MLPClassifier):
        wrapped_model = MLPWrapper(**model.get_params())
    else:
        wrapped_model = model
    
    optimizer = BayesSearchCV(
        estimator=wrapped_model,
        search_spaces=search_spaces,
        n_iter=n_iter,
        cv=3,
        n_jobs=-1,
        verbose=2
    )
    
    try:
        logging.info("Fitting the model...")
        optimizer.fit(X_train, y_train)
        logging.info("Optimization complete.")
        return optimizer.best_estimator_, optimizer.best_params_
    except Exception as e:
        logging.error(f"Error during optimization: {str(e)}")
        raise

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

# Train and evaluate models
def train_and_evaluate(X_train, X_test, y_train, y_test):
    results = []
    models_predictions = {}

    # Optimize XGBoost Model
    logging.info("Optimizing XGBoost model...")
    xgb_model, xgb_params = optimize_model(
        X_train,
        y_train,
        xgb.XGBClassifier(
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        ),
        {
            'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'max_depth': Integer(3, 8),
            'n_estimators': Integer(100, 500),
            'subsample': Real(0.7, 1.0),
            'colsample_bytree': Real(0.7, 1.0),
            'gamma': Real(0, 5),
            'reg_alpha': Real(1e-5, 1e-2, prior='log-uniform'),
            'reg_lambda': Real(1e-5, 1e-1, prior='log-uniform'),
        }
    )

    # Train XGBoost with early stopping
    logging.info("Training the best XGBoost model with early stopping...")
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

    # Optimize MLP Model
    logging.info("Optimizing MLP model...")
    mlp_model, mlp_params = optimize_model(
        X_train,
        y_train,
        MLPClassifier(
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            verbose=2
        ),
        {
            'hidden_layer_sizes': Categorical(['150,75', '300,150', '450,225']),
            'activation': Categorical(['relu']),
            'alpha': Real(1e-5, 1e-3, prior='log-uniform'),
            'learning_rate_init': Real(1e-4, 5e-3, prior='log-uniform'),
        }
    )

    # Train MLP with logging
    logging.info("Training the best MLP model...")
    mlp_model.fit(X_train, y_train)  # Verbose logging happens automatically
    mlp_result, mlp_pred = evaluate_model(mlp_model, X_test, y_test, "MLP")
    results.append(mlp_result)
    models_predictions['MLP'] = mlp_model

    # Plot Confusion Matrices
    for model_name, y_pred in models_predictions.items():
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
    results_df, models_predictions = train_and_evaluate(X_train_features, X_test_features, y_train, y_test)
    results_df.sort_values(['Accuracy', 'Precision', 'Recall', 'F1-Score'], ascending=False, inplace=True)

    # Save results
    results_df.to_csv('models_evaluation_results.csv', index=False)

        # Save the best model
    best_model = models_predictions[best_model_name]  # Retrieve the trained model
    joblib.dump(best_model, f'best_{best_model_name}.joblib')  # Save the best model

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
    custom_predictions = evaluate_custom_data(custom_news, models_predictions[best_model_name], best_tfidf)
    print("Predictions on custom dataset:", custom_predictions)
