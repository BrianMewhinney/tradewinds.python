from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pandas as pd
from io import StringIO

def model_prediction(model, X_test, threshold=0.5):
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

    X_df = pd.read_csv(StringIO(X_test), header=0)
    fixture_ids = X_df['fixtureId'].values
    X_df = X_df.drop(columns=['fixtureId'])  # Remove from features but keep IDs


    # Generate predictions
    y_proba = model.predict(X_df)
    y_pred = (y_proba >= threshold).astype(int)

    return y_pred, fixture_ids
