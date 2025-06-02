from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pandas as pd
from io import StringIO

def evaluate_model(model, X_test, y_test, threshold=0.6):
    """
    Evaluates model performance on test data

    Args:
        model: Trained LightGBM model
        X_test_csv (str): CSV string of test features
        y_test_csv (str): CSV string of test labels
        threshold (float): Classification cutoff probability

    Returns:
        dict: Evaluation metrics and confusion matrix
    """
    # Load test data
    #X_test = pd.read_csv(StringIO(X_test_csv)).astype(np.float32)
    #y_test = pd.read_csv(StringIO(y_test_csv)).squeeze().astype(int)

    # Generate predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    print(f"PROBA:{y_proba}")
    y_pred = (y_proba >= threshold).astype(int)
    #print(y_pred)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_class_0': precision_score(y_test, y_pred, pos_label=0, zero_division = 1),
        'precision_class_1': precision_score(y_test, y_pred, pos_label=1, zero_division = 1),
        'recall_class_0': recall_score(y_test, y_pred, pos_label=0, zero_division = 1),
        'recall_class_1': recall_score(y_test, y_pred, pos_label=1, zero_division = 1),
        'f1_class_0': f1_score(y_test, y_pred, pos_label=0, zero_division = 1),
        'f1_class_1': f1_score(y_test, y_pred, pos_label=1, zero_division = 1),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(
            y_test, y_pred,
            target_names=['No Draw', 'Draw']
        )
    }
    print(metrics['classification_report'])
    return metrics, y_pred, y_proba
