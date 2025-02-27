import pandas as pd
import numpy as np
from io import StringIO
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, classification_report, precision_score, recall_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import time
from datetime import datetime


def light_gbm_predictor(X_csv, y_csv):
    """
    Trains a LightGBM model for draw prediction with cross-validation

    Args:
        X_csv (str): CSV string of features (already scaled)
        y_csv (str): CSV string of target (1=draw, 0=not draw)

    Returns:
        tuple: (trained_model, feature_importances, validation_metrics)
    """
    start_time = time.time()
    results = {}
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Read CSV data from strings
    X_df = pd.read_csv(StringIO(X_csv), header=0)
    fixture_ids = X_df['fixtureId'].values
    X_df = X_df.drop(columns=['fixtureId'])  # Remove from features but keep IDs

    feature_names = np.array(X_df.columns.tolist())  # Convert to array for index access
    results['feature_names']=feature_names

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

    # LightGBM parameters
    params = {
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'boosting_type': 'gbdt',
        'num_leaves': 50,
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42,
        'is_unbalance': True  # Crucial for draw prediction imbalance
    }

    # Cross-validation setup
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    eval_results = []

    # Initialize feature importances with DataFrame columns
    feature_importances = pd.DataFrame(index=X.columns)

    all_precisions = []
    all_recalls = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train)):
        # Data partitioning (using your existing indices)
        X_fold_train = X_train.iloc[train_idx]  # DataFrame
        y_fold_train = y_train[train_idx]       # Array
        X_fold_valid = X_train.iloc[valid_idx]   # DataFrame
        y_fold_valid = y_train[valid_idx]       # Array <- This is your validation y

        # Train model on X_fold_train/y_fold_train
        # ...

        # Convert indices using .iloc on DataFrame
        train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
        valid_data = lgb.Dataset(X_fold_valid, label=y_fold_valid)

        # Model training
        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=100)
            ]
        )

        # Store feature importance
        feature_importances[f'fold_{fold+1}'] = pd.Series(
            model.feature_importance(importance_type='gain'),
            index=X.columns
        )

        # Validation predictions
        val_preds = model.predict(X_train.iloc[valid_idx])
        eval_results.append(roc_auc_score(y_train[valid_idx], val_preds))

        # Calculate metrics for THIS fold's validation data
        val_preds = model.predict(X_fold_valid)
        fold_precision = precision_score(y_fold_valid, (val_preds > 0.5).astype(int), zero_division=0)
        fold_recall = recall_score(y_fold_valid, (val_preds > 0.5).astype(int), zero_division=0)
        all_precisions.append(fold_precision)
        all_recalls.append(fold_recall)

    # Train final model on full training set
    final_model = lgb.train(
        params,
        lgb.Dataset(X_train, label=y_train),
        callbacks=[lgb.log_evaluation(period=100)]
    )

    # Evaluate on hold-out validation set
    #val_preds = final_model.predict(X_val)
    #val_metrics = {
    #    'roc_auc': roc_auc_score(y_val, val_preds),
    #    'classification_report': classification_report(
    #        y_val,
    #        (val_preds > 0.5).astype(int),
    #        target_names=['No Draw', 'Draw']
    #    )
    #}
    #print(f"Precision (Draw): {np.mean(all_precisions):.3f} ± {np.std(all_precisions):.3f}")
    #print(f"Recall (Draw): {np.mean(all_recalls):.3f} ± {np.std(all_recalls):.3f}")

    return final_model, feature_importances, X_val, y_val, id_val
