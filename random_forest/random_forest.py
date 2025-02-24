import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer, f1_score, precision_score, balanced_accuracy_score
from sklearn.inspection import permutation_importance
import time
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os
import shutil
from io import StringIO
from sklearn.model_selection import TimeSeriesSplit
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from imblearn.pipeline import Pipeline

def custom_scorer(y_true, y_pred):
    return f1_score(y_true, y_pred, average=None)[1]

class EmphasizeFeatures:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[:, 1] *= 2  # Emphasize feature 2
        return X

def random_forest_processing(x_file, y_file):
    start_time = time.time()
    results = {}
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Read CSV data from strings
    X_df = pd.read_csv(StringIO(x_file), header=0)

    # After reading X_df but before splitting
    X_df['fixtureId'] = np.arange(len(X_df))  # Create unique match identifiers
    fixture_ids = X_df['fixtureId'].values
    X_df = X_df.drop(columns=['fixtureId'])  # Remove from features but keep IDs

    corr_matrix = X_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = [column for column in upper.columns if any(upper[column] > 0.8)]
    X_df = X_df.drop(columns=high_corr)

    remaining_features = X_df.columns.tolist()  # This is critical
    feature_names = np.array(remaining_features)  # Convert to array for index access
    print(f"Remaining features ({len(remaining_features)}): {remaining_features}")
    results['feature_names']=remaining_features

    X = X_df.values
    y = pd.read_csv(StringIO(y_file), header=None).values.flatten()

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, fixture_ids,
        test_size=0.1,
        shuffle=False  # Critical for time-series order preservation
    )


    print(f"Training set size: {len(X_train)}  Test set size: {len(X_test)}")
    class_counts = np.bincount(y_train)
    print(f"Class distribution - Train: {class_counts}")
    print(f"Class distribution - Test: {np.bincount(y_test)}")

    if 0 not in np.unique(y_train) or 0 not in np.unique(y_test):
        raise ValueError("Class 0 (draw/tie) missing in train/test data")


    tscv = TimeSeriesSplit(n_splits=5)
    scorer = make_scorer(f1_score, average='binary', pos_label=1)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Optional but often helpful
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'classifier__n_estimators': [750, 1000, 1500],
        'classifier__max_depth': [5, 10, 20],
        'classifier__class_weight': [
                'balanced',
                {0: 10, 1: 1},  # Heavy emphasis on draws
                {0: class_counts[1]/class_counts[0], 1: 1}  # Auto-ratio
        ],
        'classifier__min_samples_split': [2, 5, 10, 15],
        'classifier__min_samples_leaf': [1, 5, 10],
        'classifier__max_features': ['log2', 'sqrt'],
    }
    grid_search = HalvingRandomSearchCV(
        pipeline,
        param_distributions=param_grid,
        cv=tscv,
        n_jobs=-1,
        scoring=scorer,
        #n_iter=100,
        aggressive_elimination=True,
        random_state=42
    )

    #grid_search = HalvingRandomSearchCV(RandomForestClassifier(random_state=42), param_distributions=param_grid, cv=tscv, n_jobs=-1, scoring=scorer, n_iter=100, aggressive_elimination=True)
    #grid_search = RandomizedSearchCV(n_iter=10, cv=tscv, n_jobs=-1, scoring='f1_macro', random_state=42)
    grid_search.fit(X_train, y_train)

    execution_time = time.time() - start_time
    results['start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    results['execution_time'] = execution_time
    results['best_params'] = grid_search.best_params_
    results['best_score'] = grid_search.best_score_

    best_model = grid_search.best_estimator_
    #y_pred = best_model.predict(X_test)

    probas = best_model.predict_proba(X_test)[:, 0]  # Class 0 probabilities
    optimal_threshold = 0.3  # Start with this, adjust based on precision-recall
    y_pred = (probas > optimal_threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    results['test_accuracy'] = accuracy

    cm = confusion_matrix(y_test, y_pred)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    results['per_class_accuracy'] = per_class_accuracy

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    results['balanced_accuracy'] = bal_acc
    print(f"Balanced Accuracy: {bal_acc:.4f}")

    # Calculate overall precision (macro-average)
    precision = precision_score(y_test, y_pred, average='macro')  # Replace 'macro' with other options if needed
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    results['test_precision'] = precision

    # Calculate per-class precision
    per_class_precision = precision_score(y_test, y_pred, average=None, zero_division=0)
    results['per_class_precision'] = per_class_precision

    print(f"Overall Precision: {precision:.6f} Accuracy: {accuracy:.6f}   Best Score: {grid_search.best_score_:.6f}")
    print(f"Per-Class Precision: {per_class_precision}")
    print(f"Per-Class Accuracy: {per_class_accuracy}")

    # Get feature importances
    classifier = best_model.named_steps['classifier']
    importances = classifier.feature_importances_

    indices = np.argsort(importances)[::-1]
    results['feature_importance'] = {
        feature_names[idx]: float(importances[idx])  # Store name: importance
        for idx in indices
    }

    # Permutation importance
    result = permutation_importance(
        classifier,  # Use the extracted classifier
        X_test,
        y_test,
        n_repeats=30,
        random_state=42,
        n_jobs=-1,
        scoring=scorer
    )

    sorted_idx = result.importances_mean.argsort()[::-1]
    results['permutation_importance'] = {
        feature_names[idx]: float(result.importances_mean[idx])
        for idx in sorted_idx
    }

    results['fixture_mapping'] = {
        'test_ids': id_test.tolist(),  # Array of match IDs in test set
        'predicted_labels': y_pred.tolist(),
        'true_labels': y_test.tolist()
    }

    # Convert arrays to CSV strings
    y_pred_csv = StringIO()
    y_test_csv = StringIO()
    X_test_csv = StringIO()

    np.savetxt(y_pred_csv, y_pred, delimiter=",");
    np.savetxt(y_test_csv, y_test, delimiter=",");
    # Add match IDs as first column in X_test CSV
    X_test_with_ids = np.column_stack([id_test, X_test])
    np.savetxt(X_test_csv, X_test_with_ids, delimiter=",",
               header="fixture_id," + ",".join(feature_names), comments='')

    y_pred_csv.seek(0)
    y_test_csv.seek(0)
    X_test_csv.seek(0)

    print(f"Execution Time: {time.time() - start_time:.2f} seconds\n")
    return y_pred_csv.getvalue(), y_test_csv.getvalue(), X_test_csv.getvalue(), results, importances
