from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
import pandas as pd
import numpy as np
from io import StringIO

def evaluate_model(fold_models, X_test, y_test, threshold=0.32):
    """
    Evaluates model performance on test data

    Args:
        fold_models: Stratified models
        X_test_csv (str): CSV string of test features
        y_test_csv (str): CSV string of test labels
        threshold (float): Classification cutoff probability

    Returns:
        dict: Evaluation metrics and confusion matrix
    """

    # Generate predictions
    y_proba = np.zeros(len(X_test))  # Array for final ensemble predictions
    for fold_model in fold_models:
        fold_proba = fold_model.predict_proba(X_test)[:, 1]  # Add probabilities
        #print(fold_proba)
        y_proba += fold_proba
    y_proba /= len(fold_models)  # Average the proba
    y_pred = (y_proba >= threshold).astype(int)

    # Calculate metrics
    metrics = {
        'roc_auc_score': roc_auc_score(y_test, y_proba),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_class_0': precision_score(y_test, y_pred, pos_label=0, zero_division=1),
        'precision_class_1': precision_score(y_test, y_pred, pos_label=1, zero_division=1),
        'recall_class_0': recall_score(y_test, y_pred, pos_label=0, zero_division = 1),
        'recall_class_1': recall_score(y_test, y_pred, pos_label=1, zero_division = 1),
        'f1_class_0': f1_score(y_test, y_pred, pos_label=0, zero_division = 1),
        'f1_class_1': f1_score(y_test, y_pred, pos_label=1, zero_division = 1),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(
            y_test, y_pred,
            target_names=['No Draw', 'Draw'],
            zero_division=1
        )
    }
    #print(f"ROC-AUC: {metrics['roc_auc_score']}")
    return metrics, y_pred, y_proba
