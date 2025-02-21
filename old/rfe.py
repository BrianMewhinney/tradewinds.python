import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFECV
import time
from datetime import datetime
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

def rfe_processing(x_file, y_file):
    start_time = time.time()
    results = {}
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Read the data with head
    X_df = pd.read_csv(x_file, header=0)
    feature_names = X_df.columns  # Extract feature names
    X = X_df.values

    y = pd.read_csv(y_file, header=None).values.flatten()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    print(f"Training set size: {len(X_train)}  Test set size: {len(X_test)}")

    model = RandomForestClassifier(random_state=42, n_jobs=-1)

    # Define the parameter grid
    param_grid = {
        'n_estimators': [750, 1000, 1500, 2000],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 5, 10],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', None]
    }
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Set up RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=10,
        cv=stratified_kfold,
        random_state=42,
        n_jobs=-1
    )

    # Fit RandomizedSearchCV
    random_search.fit(X_train, y_train)

    # Use the best estimator found by RandomizedSearchCV
    best_model = random_search.best_estimator_

    print(f"Best Parameters: {random_search.best_params_}")
    y_pred = best_model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):5f}")


    # Perform RFECV using the best model
    rfecv = RFECV(estimator=best_model, step=1, cv=5, scoring='accuracy', n_jobs=-1)
    rfecv.fit(X_train, y_train)

    execution_time = time.time() - start_time

    rankings = rfecv.ranking_
    support = rfecv.support_

    print(f"Optimal number of features: {rfecv.n_features_}")

    feature_info = [(feature_names[i], rank, selected) for i, (rank, selected) in enumerate(zip(rankings, support))]

    sorted_features = sorted(feature_info, key=lambda x: x[1])  # Sort by rank (ascending)

    print("Selected Features (ranking = 1):")
    for feature_name, rank, selected in sorted_features:
        if selected:
            print(f"Feature {feature_name}: Ranking = {rank}, Selected = {selected}")

    correlation_matrix = X_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.title('Feature Correlation Matrix')
    plt.show()

    high_corr_pairs = [(i, j) for i in correlation_matrix.columns for j in correlation_matrix.columns if i != j and abs(correlation_matrix[i][j]) > 0.8]

    print("Highly correlated feature pairs:")
    for pair in high_corr_pairs:
        print(pair)

    print(f"End Time.  Total: {execution_time} seconds)")
    return rfecv.ranking_, rfecv.support_
