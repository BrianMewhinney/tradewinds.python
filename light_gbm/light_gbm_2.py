import pandas as pd
import numpy as np
from io import StringIO
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, classification_report, precision_score, recall_score
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
import time
from datetime import datetime

def light_gbm_predictor(X_csv, y_csv):
    start_time = time.time()
    results = {}
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Read CSV data from strings
    X_df = pd.read_csv(StringIO(X_csv), header=0)
    fixture_ids = X_df['fixtureId'].values
    X_df = X_df.drop(columns=['fixtureId'])

    feature_names = np.array(X_df.columns.tolist())
    results['feature_names'] = feature_names

    X = X_df
    y = pd.read_csv(StringIO(y_csv), header=None).values.flatten()

    # Convert to proper formats
    X = X.astype(np.float32)
    y = y.astype(int)

    # Split initial data
    X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
        X, y, fixture_ids,
        test_size=.1,
        stratify=y,
        random_state=42
    )

    # Define parameter grid
    param_grid = {
        'num_leaves': [31, 50, 70],
        'learning_rate': [0.01, 0.05, 0.1],
        'feature_fraction': [0.7, 0.9, 1.0],
        'bagging_fraction': [0.7, 0.8, 0.9],
        'bagging_freq': [5, 10],
        'is_unbalance': [True, False]
    }

    # Initialize LightGBM classifier
    lgb_estimator = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        boosting_type='gbdt',
        verbose=-1,
        seed=42
    )

    # Setup cross-validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Setup RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=lgb_estimator,
        param_distributions=param_grid,
        n_iter=10,
        scoring='roc_auc',
        cv=skf,
        random_state=42,
        verbose=1,
        n_jobs=-1
    )

    # Fit RandomizedSearchCV
    random_search.fit(X_train, y_train)

    # Best model from random search
    best_model = random_search.best_estimator_

    # Train final model on full training set
    final_model = best_model.fit(X_train, y_train)

    # Evaluate on hold-out validation set
    val_preds = final_model.predict_proba(X_val)[:, 1]
    val_metrics = {
        'roc_auc': roc_auc_score(y_val, val_preds),
        'classification_report': classification_report(
            y_val,
            (val_preds > 0.5).astype(int),
            target_names=['No Draw', 'Draw']
        )
    }

    print(f"Best Parameters: {random_search.best_params_}")
    print(f"ROC AUC on Validation Set: {val_metrics['roc_auc']:.3f}")
    print(val_metrics['classification_report'])

    return final_model, random_search.best_params_, X_val, y_val, id_val

# Example usage:
# final_model, best_params, X_val, y_val, id_val = light_gbm_predictor(X_csv, y_csv)
