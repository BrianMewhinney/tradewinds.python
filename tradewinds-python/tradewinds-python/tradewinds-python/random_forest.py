import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
import time
from datetime import datetime

def random_forest_processing(x_file, y_file):
    start_time = time.time()
    results = {}
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    X = pd.read_csv(x_file).values
    y = pd.read_csv(y_file).values.flatten()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#    print(f"Training set size: {len(X_train)}")
#    print(f"Test set size: {len(X_test)}")

    param_grid = {
        'n_estimators': [500, 750, 1000, 1500],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 5, 10],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', None]
    }

    grid_search = RandomizedSearchCV(RandomForestClassifier(random_state=42),
                                     param_distributions=param_grid,
                                     n_iter=50, cv=3, n_jobs=-1,
                                     random_state=42)
    grid_search.fit(X_train, y_train)  # Use the original y_train

    execution_time = time.time() - start_time
    results['start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    results['execution_time'] = execution_time
    results['best_params'] = grid_search.best_params_
    results['best_score'] = grid_search.best_score_

#    print(f'Best Score: {grid_search.best_score_:.6f}')
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)  # Standardize X_test
    accuracy = accuracy_score(y_test, y_pred)
    results['test_accuracy'] = accuracy

    # Feature importance
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    results['feature_importance'] = {f"{f + 1}": (indices[f], importances[indices[f]]) for f in range(X_train.shape[1])}

    # Permutation importance
    result = permutation_importance(best_model, X_test, y_test, n_repeats=30,
                                    random_state=42, n_jobs=-1)

    sorted_idx = result.importances_mean.argsort()[::-1]
    results['permutation_importance'] = {idx: result.importances_mean[idx] for idx in sorted_idx}
    print(f"Execution Time: {time.time() - start_time:.2f} seconds")

    return results, importances
