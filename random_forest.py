import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer, f1_score, precision_score
from sklearn.inspection import permutation_importance
import time
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os
import shutil
from imblearn.over_sampling import SMOTE

def custom_scorer(y_true, y_pred):
    return f1_score(y_true, y_pred, average=None)[0]

class EmphasizeFeatures:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[:, 0] *= 4  # Emphasize feature 0
        X[:, 3] *= 2  # Emphasize feature 3
        return X

def random_forest_processing(x_file, y_file):
    start_time = time.time()
    results = {}
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    X_df = pd.read_csv(x_file, header=0)
    if 'w_diffcategory' in X_df.columns:
        X_df['w_diffcategory'] = X_df['w_diffcategory'].astype('category')
        # Use one-hot encoding
        X_df = pd.get_dummies(X_df, columns=['w_diffcategory'], prefix='sig')

    # Update feature names after one-hot encoding
    feature_names = X_df.columns

    X = X_df.values
    y = pd.read_csv(y_file, header=None).values.flatten()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
    print(f"Training set size: {len(X_train)}  Test set size: {len(X_test)}")

    # Compute class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}

    param_grid = {
#        'classifier__n_estimators': [750, 1000, 1500, 2000],
#        'classifier__max_depth': [5, 10, 20],
#        'classifier__min_samples_split': [2, 5, 10, 15],
#        'classifier__min_samples_leaf': [1, 5, 10],
#        'classifier__max_features': ['log2', 'sqrt'],
#        'classifier__class_weight': [None, 'balanced']
        'n_estimators': [750, 1000, 1500, 2000],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 5, 10],
        'max_features': ['log2', 'sqrt'],
        'class_weight': [None, 'balanced']
        #'class_weight': [class_weight_dict]
    }

    #pipeline = Pipeline([
    #    ('emphasize', EmphasizeFeatures()),
    #    ('scaler', StandardScaler()),
    #    ('classifier', RandomForestClassifier(random_state=42))
    #])
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    #grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=stratified_kfold, scoring='f1_macro', n_jobs=-1)
    grid_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_distributions=param_grid, n_iter=10, cv=stratified_kfold, n_jobs=-1, scoring='f1_macro', random_state=42)
    #grid_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_distributions=param_grid, n_iter=10, cv=stratified_kfold, n_jobs=-1, scoring=make_scorer(custom_scorer), random_state=42)
    #grid_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=10, cv=stratified_kfold, n_jobs=-1, scoring='f1_macro', random_state=42)
    grid_search.fit(X_train_resampled, y_train_resampled)

    execution_time = time.time() - start_time
    results['start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    results['execution_time'] = execution_time
    results['best_params'] = grid_search.best_params_
    results['best_score'] = grid_search.best_score_

#    print(f"Best Parameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    best_model.fit(X_train_resampled, y_train_resampled)

    y_pred = best_model.predict(X_test)  # Standardize X_test

    accuracy = accuracy_score(y_test, y_pred)
    results['test_accuracy'] = accuracy

    # Calculate per-class accuracy
    cm = confusion_matrix(y_test, y_pred)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    results['per_class_accuracy'] = per_class_accuracy

    # Calculate overall precision (macro-average)
    precision = precision_score(y_test, y_pred, average='macro')  # Replace 'macro' with other options if needed
    results['test_precision'] = precision
    # Calculate per-class precision
    per_class_precision = precision_score(y_test, y_pred, average=None)  # Precision for each class
    results['per_class_precision'] = per_class_precision

    print(f"Overall Precision: {precision:.6f} Accuracy: {accuracy:.6f}   Best Score: {grid_search.best_score_:.6f}")
    print(f"Per-Class Precision: {per_class_precision}")
    print(f"Per-Class Accuracy: {per_class_accuracy}")

    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    results['feature_importance'] = {f"{f + 1}": (indices[f], importances[indices[f]]) for f in range(X_train_resampled.shape[1])}
    for idx in indices:
        print(f"{idx}: {feature_names[idx]}: {importances[idx]}")

    result = permutation_importance(best_model, X_test, y_test, n_repeats=30, random_state=42, n_jobs=-1)

    sorted_idx = result.importances_mean.argsort()[::-1]
    print('')
    for idx in sorted_idx:
        print(f"{idx}: {feature_names[idx]}: {result.importances_mean[idx]}")

    results['permutation_importance'] = {idx: result.importances_mean[idx] for idx in sorted_idx}
    output_dir = os.path.dirname(x_file)
    output_dir = output_dir.replace("input", "output")
    os.makedirs(output_dir, exist_ok=True)
    print(f'Saving files to {output_dir}')
    np.savetxt(os.path.join(output_dir, "y_pred.csv"), y_pred, delimiter=",")
    np.savetxt(os.path.join(output_dir, "y_test.csv"), y_test, delimiter=",")
    np.savetxt(os.path.join(output_dir, "X_test.csv"), X_test, delimiter=",", header=",".join(feature_names), comments='')
    shutil.copy(x_file, os.path.join(output_dir, "x.csv"))
    print(f"Execution Time: {time.time() - start_time:.2f} seconds\n")

    return results, importances
