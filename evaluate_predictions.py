from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score
import pandas as pd
import numpy as np
from io import StringIO

def evaluate_predictions(fold_models, X_csv, y_csv):
    # Transform the data first
    X_df = pd.read_csv(StringIO(X_csv), header=0)
    fixture_ids = X_df['fixtureId'].values
    X_df = X_df.drop(columns=['fixtureId'])  # Remove from features but keep IDs
    feature_names = np.array(X_df.columns.tolist())  # Convert to array for index access
    X = X_df
    X = X.astype(np.float32)
    y = pd.read_csv(StringIO(y_csv), header=None).values.flatten()
    y = y.astype(int)

    # Generate predictions
    y_proba = np.zeros(len(X))  # Array for final ensemble predictions
    for fold_model in fold_models:
        fold_proba = fold_model.predict_proba(X)[:, 1]  # Add probabilities
        y_proba += fold_proba

    y_proba /= len(fold_models)  # Average the proba
    print(f"Prediction Probability: : {y_proba}")

    return y_proba, fixture_ids, y
