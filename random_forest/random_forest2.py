import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer, f1_score, precision_score, balanced_accuracy_score, recall_score, precision_recall_curve
from sklearn.inspection import permutation_importance
import time
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
import os
import shutil
from io import StringIO
from sklearn.model_selection import TimeSeriesSplit
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from imblearn.pipeline import Pipeline
import gc

#np.seterr(all='raise')

def safe_draw_scorer(y_true, y_pred):
    """Handles cases where class 0 is missing"""
    if 0 not in y_true:
        return 0.0  # No draws to evaluate
    return f1_score(y_true, y_pred, pos_label=0, zero_division=0)

def random_forest_processing(x_file, y_file):
    start_time = time.time()
    results = {}
    gc.collect()

    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Read CSV data from strings
    X_df = pd.read_csv(StringIO(x_file), header=0)
    # After reading X_df but before splitting
    fixture_ids = X_df['fixtureId'].values
    X_df = X_df.drop(columns=['fixtureId'])  # Remove from features but keep IDs

    feature_names = np.array(X_df.columns.tolist())  # Convert to array for index access
    print(f"Feature Names:({feature_names})")
    results['feature_names']=feature_names
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
    draw_ratio = class_counts[0] / len(y_train)
    print(f"Draw percentage: {draw_ratio:.2%}")
    if 0 not in np.unique(y_train) or 0 not in np.unique(y_test):
        raise ValueError("Class 0 (draw/tie) missing in train/test data")

    tscv = TimeSeriesSplit(n_splits=5)
    draw_scorer = make_scorer(safe_draw_scorer)

    pipeline = Pipeline([# Optional but often helpful
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'classifier__n_estimators': [750, 1000, 1500],
        'classifier__max_depth': [5, 10, 20],
        'classifier__class_weight': [
            {0: 5, 1: 1},  # Heavy draw emphasis
            #{0: 2, 1: 1},
            #'balanced'
        ],
        'classifier__min_samples_split': [2, 5, 10, 15],
        'classifier__min_samples_leaf': [1, 5, 10],
        'classifier__max_features': ['log2', 'sqrt'],
    }
    scorer = draw_scorer
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

    #grid_search = RandomizedSearchCV(n_iter=10, cv=tscv, n_jobs=-1, scoring='f1_macro', random_state=42)
    grid_search.fit(X_train, y_train)

    execution_time = time.time() - start_time
    results['start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    results['execution_time'] = execution_time
    results['best_params'] = grid_search.best_params_
    results['best_score'] = grid_search.best_score_

    best_model = grid_search.best_estimator_

    # Calculate metrics at 0.5 threshold (standard predictions)
    y_pred = best_model.predict(X_test)

    # Calculate metrics normally
    results['draw_metrics'] = {
        'precision': precision_score(y_test, y_pred, pos_label=0, zero_division=0),
        'recall': recall_score(y_test, y_pred, pos_label=0, zero_division=0),
        'f1': f1_score(y_test, y_pred, pos_label=0, zero_division=0)
    }

    print(f"Draw F1: {results['draw_metrics']['f1']:.2%}")
    print(f"Draw Precision: {results['draw_metrics']['precision']:.2%}")
    print(f"Draw Recall: {results['draw_metrics']['recall']:.2%}")

    cm = confusion_matrix(y_test, y_pred)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    results['per_class_accuracy'] = per_class_accuracy

    # Calculate per-class precision
    per_class_precision = precision_score(y_test, y_pred, average=None, zero_division=0)
    results['per_class_precision'] = per_class_precision
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
    print(results['feature_importance'])

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
    print(results['permutation_importance'])

    results['fixture_mapping'] = {
        'test_ids': id_test.tolist(),  # Array of match IDs in test set
        'predicted_labels': y_pred.tolist(),
        'true_labels': y_test.tolist()
    }

    # Convert arrays to CSV strings
    y_pred_csv = StringIO()
    y_test_csv = StringIO()
    X_test_csv = StringIO()

    np.savetxt(y_pred_csv, y_pred, delimiter=",")
    np.savetxt(y_test_csv, y_test, delimiter=",")
    # Add match IDs as first column in X_test CSV
    X_test_with_ids = np.column_stack([id_test, X_test])
    np.savetxt(X_test_csv, X_test_with_ids, delimiter=",",
               header="fixture_id," + ",".join(feature_names), comments='')

    y_pred_csv.seek(0)
    y_test_csv.seek(0)
    X_test_csv.seek(0)

    print(f"Execution Time: {time.time() - start_time:.2f} seconds\n")
    return y_pred_csv.getvalue(), y_test_csv.getvalue(), X_test_csv.getvalue(), results, importances
