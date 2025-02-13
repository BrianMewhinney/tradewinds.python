import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix,  make_scorer, f1_score
from sklearn.inspection import permutation_importance
import time
from datetime import datetime
import os

def custom_scorer(y_true, y_pred):
    return f1_score(y_true, y_pred, average=None)[0]

def xgboost_processing(x_file, y_file):
    start_time = time.time()
    results = {}
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    X_df = pd.read_csv(x_file, header=0)
    feature_names = X_df.columns  # Extract feature names
    X = X_df.values

    y = pd.read_csv(y_file, header=None).values.flatten()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    print(f"Training set size: {len(X_train)}  Test set size: {len(X_test)}")

    param_grid = {
        'n_estimators': [750, 1000, 1500, 2000],
        'max_depth': [5, 10, 20],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2]
    }
    stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    #grid_search = GridSearchCV(XGBClassifier(eval_metric='mlogloss', random_state=42), param_grid, cv=stratified_kfold, n_jobs=-1, scoring='f1_macro')
    #grid_search = RandomizedSearchCV(XGBClassifier(eval_metric='mlogloss', random_state=42), param_distributions=param_grid, n_iter=10, cv=stratified_kfold, n_jobs=-1, scoring='f1_macro', random_state=42)
    grid_search = RandomizedSearchCV(XGBClassifier(eval_metric='mlogloss', random_state=42), param_distributions=param_grid, n_iter=10, cv=stratified_kfold, n_jobs=-1, scoring=make_scorer(custom_scorer), random_state=42)
    grid_search.fit(X_train, y_train)

    execution_time = time.time() - start_time
    results['start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    results['execution_time'] = execution_time
    results['best_params'] = grid_search.best_params_
    results['best_score'] = grid_search.best_score_

    #print(f'Best Score: {grid_search.best_score_:.6f}')
    #print(f"Best Parameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results['test_accuracy'] = accuracy

    # Calculate per-class accuracy
    cm = confusion_matrix(y_test, y_pred)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    results['per_class_accuracy'] = per_class_accuracy

    print(f"Overall Test Accuracy: {accuracy:.6f}   Best Score: {grid_search.best_score_:.6f}")
    print(f"Per-Class Accuracy: {per_class_accuracy}")

    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    results['feature_importance'] = {f"{f + 1}": (indices[f], importances[indices[f]]) for f in range(X_train.shape[1])}
    for idx in indices:
        print(f"{idx}: {feature_names[idx]}: {importances[idx]}")

    result = permutation_importance(best_model, X_test, y_test, n_repeats=30, random_state=42, n_jobs=-1)

    sorted_idx = result.importances_mean.argsort()[::-1]
    print('')
    for idx in sorted_idx:
        print(f"{idx}: {feature_names[idx]}: {result.importances_mean[idx]}")

    results['permutation_importance'] = {idx: result.importances_mean[idx] for idx in sorted_idx}
    #indices = np.argsort(importances)[::-1]
    #results['feature_importance'] = {f"{f}: {feature_names[f]}": (indices[f], importances[indices[f]]) for f in indices}
    #results['feature_importance'] = {f"{feature_names[f]}": importances[indices[f]] for f in range(X_train.shape[1])}
    #print("Feature Importances:")
    #for i in indices:
        #print(f"{feature_names[i]}: {importances[i]}")

    #result = permutation_importance(best_model, X_test, y_test, n_repeats=30, random_state=42, n_jobs=-1)

    #sorted_idx = result.importances_mean.argsort()[::-1]
    #print("\nPermutation Importances:")
    #for idx in sorted_idx:
        #print(f"{feature_names[idx]}: {result.importances_mean[idx]}")

    #results['permutation_importance'] = {f"{feature_names[idx]}": result.importances_mean[idx] for idx in sorted_idx}

    output_dir = os.path.dirname(x_file)
    np.savetxt(os.path.join(output_dir, "y_pred.csv"), y_pred, delimiter=",")
    np.savetxt(os.path.join(output_dir, "X_test.csv"), X_test, delimiter=",", header=",".join(feature_names), comments='')
    print(f"Execution Time: {time.time() - start_time:.2f} seconds")

    return results, importances
