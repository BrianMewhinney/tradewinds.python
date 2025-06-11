import pandas as pd
import numpy as np
from io import StringIO
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, classification_report, precision_score, recall_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.inspection import permutation_importance
import shap
import time
from datetime import datetime

def light_gbm_predictor(X_csv, y_csv, PredX_csv):
    start_time = time.time()
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Read CSV data from strings
    X_df = pd.read_csv(StringIO(X_csv), header=0)
    fixture_ids = X_df['fixtureId'].values
    X_df = X_df.drop(columns=['fixtureId'])  # Remove from features but keep IDs
    feature_names = np.array(X_df.columns.tolist())  # Convert to array for index access
    X = X_df
    y = pd.read_csv(StringIO(y_csv), header=None).values.flatten()

    # Convert to proper formats
    print(X)
    X = X.astype(np.float32)
    y = y.astype(int)

    # Determine the split index
    split_index = int(len(X) - (len(X) * 0.2))
    print(f"Split Index {split_index} of {len(X)}")

    # Split data into train and validation sets
    X_train, X_val = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]
    id_train, id_val = fixture_ids[:split_index], fixture_ids[split_index:]

    #pd.set_option('display.max_columns', None)
    #print(X_train.describe())
    # Optionally reset to default after
    #pd.reset_option('display.max_columns')

    # Check if PredX_csv has sufficient length before processing
    if len(PredX_csv.strip()) > 0:
        print("GREATER THAN 0")
        PredX_df = pd.read_csv(StringIO(PredX_csv), header=0)
        PredX_fixture_ids = PredX_df['fixtureId'].values
        PredX_df = PredX_df.drop(columns=['fixtureId'])  # Remove from features but keep IDs

        # Ensure PredX_df has the same columns as X_df
        PredX_df = PredX_df[feature_names]

        # Convert to proper format
        PredX = PredX_df.astype(np.float32)

        # Append PredX to X_val and PredX_fixture_ids to id_val
        X_val = pd.concat([X_val, PredX_df], ignore_index=True)
        id_val = np.concatenate((id_val, PredX_fixture_ids))
        y_val = np.concatenate((y_val, np.zeros(len(PredX_df))))

    # LightGBM parameters
    params = {
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'boosting_type': 'gbdt',
        'num_leaves': 64,
        'learning_rate': 0.03,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42,
        'n_jobs': -1,
        'n_estimators': 1000,
        'is_unbalance': True  # Crucial for draw prediction imbalance
    }

    # Cross-validation setup
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(X_train))  # OOF probabilities
    oof_true = np.zeros(len(X_train))   # OOF true labels
    oof_fixture_ids = np.empty(len(X_train), dtype=fixture_ids.dtype)
    fold_models = []

    # Initialize feature importance and permutation importance with DataFrame columns
    feature_importances = pd.DataFrame(index=X.columns)
    perm_importances = pd.DataFrame(index=X.columns)
    fold_auc_scores = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train)):
        # Data partitioning (using your existing indices)
        X_fold_train = X_train.iloc[train_idx]  # DataFrame
        y_fold_train = y_train[train_idx]       # Array
        X_fold_valid = X_train.iloc[valid_idx]  # DataFrame
        y_fold_valid = y_train[valid_idx]       # Array

        # Train model on X_fold_train/y_fold_train
        model = LGBMClassifier(**params)
        model.fit(
            X_fold_train,
            y_fold_train,
            eval_set=[(X_fold_valid, y_fold_valid)],
            eval_metric=['binary_logloss', 'auc'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=80, verbose=False),
                #lgb.log_evaluation(period=50)
            ]
        )
        print(model.best_iteration_)
        booster = model.booster_
        dumped = booster.dump_model()

        # Each tree is a dict in dumped['tree_info']
        leaves_per_tree = [tree['num_leaves'] for tree in dumped['tree_info']]
        max_leaves_used = max(leaves_per_tree)
        min_leaves_used = min(leaves_per_tree)
        avg_leaves_used = sum(leaves_per_tree) / len(leaves_per_tree)

        print(f"Maximum leaves used in any tree: {max_leaves_used}")
        print(f"Minimum leaves used in any tree: {min_leaves_used}")
        print(f"Average leaves per tree: {avg_leaves_used:.2f}")

        fold_models.append(model)

        # Store OOF predictions for this fold
        valid_pred_proba = model.predict_proba(X_train.iloc[valid_idx])[:, 1]
        oof_preds[valid_idx] = valid_pred_proba
        oof_true[valid_idx] = y_train[valid_idx]
        oof_fixture_ids[valid_idx] = fixture_ids[valid_idx]

        fold_auc = roc_auc_score(y_fold_valid, valid_pred_proba)
        print(f"Fold {fold+1} ROC AUC: {fold_auc:.4f}")
        fold_auc_scores.append(fold_auc)

        # Store feature importance
        feature_importances[f'fold_{fold+1}'] = pd.Series(
            model.feature_importances_,
            index=X.columns
        )

        # Store permutation importance
        perm_importance = permutation_importance(
            model,
            X_fold_valid,
            y_fold_valid,
            n_repeats=10,
            random_state=42,
            scoring='roc_auc'
        )
        perm_importances[f'fold_{fold+1}'] = pd.Series(
            perm_importance.importances_mean,
            index=X.columns
        )

    mean_auc = np.mean(fold_auc_scores)
    print(f"Mean ROC AUC across folds: {mean_auc:.4f}")

    # Return updated results
    return {
        'fold_models': fold_models,
        'oof_preds': oof_preds,
        'oof_true': oof_true,
        'oof_fixture_ids': oof_fixture_ids,
        'feature_importances': feature_importances,
        'perm_importances': perm_importances,
        'X_val': X_val,
        'y_val': y_val,
        'id_val': id_val,
        'fold_auc_scores': fold_auc_scores,
        'mean_auc': mean_auc,

        #'final_model': final_model,
        #'shap_values': shap_values,
        #'shap_expected_value': shap_expected_value,
        #'shap_summary_df': shap_summary_df
    }



    # Train final model on full training set
    #final_model = LGBMClassifier(**params)
    #final_model.fit(X_train, y_train)

    # Compute SHAP values
    #print(f"Before SHAP Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    #explainer = shap.TreeExplainer(final_model, data=X_train)
    #shap_values_raw = explainer(X_val.sample(min(100, len(X_val))), check_additivity=False)
    #shap_expected_value = explainer.expected_value
    #print(f'Shap Expected Value: {shap_expected_value}')
    #shap_values = shap_values_raw.values.tolist()
    #shap_values_array = np.array(shap_values)

    # Calculate mean absolute SHAP values across all validation samples
    #mean_abs_shap = np.abs(shap_values_array).mean(axis=0)

    # Create sorted DataFrame
    #shap_summary_df = pd.DataFrame({
    #    'feature': X_val.columns,
    #    'mean_abs_shap': mean_abs_shap
    #}).sort_values('mean_abs_shap', ascending=False)
    #print(f"After SHAP Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Calculate permutation importance
    #perm_importance = permutation_importance(
    #    final_model,
    #    X_val,
    #    y_val,
    #    n_repeats=10,
    #    random_state=42,
    #    scoring='roc_auc'
    #)

    # Create a DataFrame for permutation importance
    #perm_importance_df = pd.DataFrame({
    #    'feature': X_val.columns,
    #    'importance_mean': perm_importance.importances_mean,
    #    'importance_std': perm_importance.importances_std
    #}).sort_values('importance_mean', ascending=False)
