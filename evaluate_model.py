from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score
import pandas as pd
import numpy as np
from io import StringIO

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def evaluate_model(fold_models, X_test, y_test, zM_test, threshold):
    y_proba = np.zeros(len(X_test), dtype=float)
    for booster in fold_models:
        r_hat = booster.predict(X_test, raw_score=True)  # residual logits
        y_proba += sigmoid(zM_test + r_hat)
    y_proba /= len(fold_models)
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        'roc_auc_score': roc_auc_score(y_test, y_proba),
        'pr_roc_score': average_precision_score(y_test, y_proba),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_class_0': precision_score(y_test, y_pred, pos_label=0, zero_division=1),
        'precision_class_1': precision_score(y_test, y_pred, pos_label=1, zero_division=1),
        'recall_class_0': recall_score(y_test, y_pred, pos_label=0, zero_division=1),
        'recall_class_1': recall_score(y_test, y_pred, pos_label=1, zero_division=1),
        'f1_class_0': f1_score(y_test, y_pred, pos_label=0, zero_division=1),
        'f1_class_1': f1_score(y_test, y_pred, pos_label=1, zero_division=1),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(
            y_test, y_pred, target_names=['No Draw', 'Draw'], zero_division=1
        )
    }
    return metrics, y_pred, y_proba

