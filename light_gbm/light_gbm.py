import pandas as pd
import numpy as np
from io import StringIO
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, classification_report, precision_score, recall_score, average_precision_score, log_loss
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.inspection import permutation_importance
import shap
import time
from datetime import datetime
from typing import Tuple, Callable, Dict, Any
from sklearn.linear_model import LogisticRegression

def logit(p):
    p = np.clip(p.astype(float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def fit_platt_calibrator(oof_probs: np.ndarray, oof_true: np.ndarray):
    # Same as your original Platt on final probabilities
    eps = 1e-6
    p = np.clip(oof_probs.astype(float), eps, 1 - eps)
    z = np.log(p / (1 - p))
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(z.reshape(-1, 1), oof_true.astype(int))
    A = float(lr.coef_[0, 0])
    B = float(lr.intercept_[0])
    def calibrate(p_new: np.ndarray) -> np.ndarray:
        p_new = np.clip(p_new.astype(float), eps, 1 - eps)
        z_new = np.log(p_new / (1 - p_new))
        logits = A * z_new + B
        return 1.0 / (1.0 + np.exp(-logits))
    return A, B, calibrate

class BoosterWithMarketOffset:
    def __init__(self, booster, zM_valid):
        self.booster = booster
        self.zM_valid = zM_valid  # per-row market logit for the validation slice
        self._estimator_type = "classifier"    # key: tell sklearn it's a classifier
        self.classes_ = np.array([0, 1])       # helps some sklearn utilities

    def fit(self, X=None, y=None):
        return self

    def predict_proba(self, X):
        # raw residual from trees
        r_hat = self.booster.predict(X, raw_score=True)
        z_total = self.zM_valid + r_hat
        p = sigmoid(z_total)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def light_gbm_predictor(X_csv, y_csv, PredX_csv):
    start_time = time.time()
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Read CSV data from strings
    X_df_all = pd.read_csv(StringIO(X_csv), header=0)

    if 'pMkt' not in X_df_all.columns:
        raise ValueError("X_csv must contain a column named 'pMkt' with the de-vig market probability for the target.")
    fixture_ids = X_df_all['fixtureId'].values
    pMkt_all = X_df_all['pMkt'].astype(float).values
    zM_all = logit(pMkt_all)

    # Remove market prob and id from model features
    X_df = X_df_all.drop(columns=['fixtureId', 'pMkt'])
    feature_names = np.array(X_df.columns.tolist())

    y = pd.read_csv(StringIO(y_csv), header=None).values.flatten().astype(int)

    # Convert to proper formats
    X = X_df.astype(np.float32).reset_index(drop=True)
    zM_all = zM_all.astype(np.float32)
    fixture_ids = np.array(fixture_ids)

    # Determine the split index
    split_index = int(len(X) - (len(X) * 0.1))
    print(f"Split Index {split_index} of {len(X)}")

    # Split data into train and validation sets
    X_train, X_val = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]
    zM_train, zM_val = zM_all[:split_index], zM_all[split_index:]
    id_train, id_val = fixture_ids[:split_index], fixture_ids[split_index:]

    # LightGBM parameters
    params = {
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'boosting_type': 'gbdt',
        'num_leaves': 128,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42,
        'num_threads': -1,
        'is_unbalance': True
    }
    num_boost_round = 2000
    early_stopping_rounds = 80

    # Walk-forward (expanding window) cross-validation setup
    n_splits = 8
    fold_size = (len(X_train) // (n_splits + 1))
    print(f'Fold size: {fold_size}')
    fold_auc_scores = []
    fold_pr_auc_scores = []
    fold_models = []

    feature_importances = pd.DataFrame(index=X.columns)
    perm_importances = pd.DataFrame(index=X.columns)
    shap_importances = pd.DataFrame(index=X.columns)

    oof_preds = np.zeros(len(X_train))  # OOF final probabilities
    oof_true = np.zeros(len(X_train))   # OOF true labels
    oof_fixture_ids = np.empty(len(X_train), dtype=fixture_ids.dtype)
    oof_folds = []

    for fold in range(n_splits):
        train_end = (fold + 1) * fold_size
        val_start = train_end
        val_end = min(val_start + fold_size, len(X_train))
        if val_start >= len(X_train):
            break

        X_fold_train = X_train.iloc[:train_end]
        y_fold_train = y_train[:train_end]
        zM_fold_train = zM_train[:train_end]

        X_fold_valid = X_train.iloc[val_start:val_end]
        y_fold_valid = y_train[val_start:val_end]
        zM_fold_valid = zM_train[val_start:val_end]

        # LightGBM Datasets with init_score = zM (market log-odds) so trees learn residuals
        dtrain = lgb.Dataset(
            X_fold_train,
            label=y_fold_train,
            init_score=zM_fold_train  # offset
        )
        dvalid = lgb.Dataset(
            X_fold_valid,
            label=y_fold_valid,
            init_score=zM_fold_valid,
            reference=dtrain
        )

        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dvalid],
            valid_names=['valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                # lgb.log_evaluation(period=50)
            ]
        )
        print(f"Fold {fold+1} best iteration: {booster.best_iteration}")
        fold_models.append(booster)

        dumped = booster.dump_model()
        leaves_per_tree = [tree['num_leaves'] for tree in dumped['tree_info']]
        print(f"Maximum leaves used in any tree: {max(leaves_per_tree)}")
        print(f"Minimum leaves used in any tree: {min(leaves_per_tree)}")
        print(f"Average leaves per tree: {sum(leaves_per_tree)/len(leaves_per_tree):.2f}")

        # OOF predictions for this fold (use residual + zM)
        r_hat_valid = booster.predict(X_fold_valid, raw_score=True)
        z_total_valid = zM_fold_valid + r_hat_valid
        valid_pred_proba = sigmoid(z_total_valid)

        oof_preds[val_start:val_end] = valid_pred_proba
        oof_true[val_start:val_end] = y_fold_valid
        oof_fixture_ids[val_start:val_end] = fixture_ids[val_start:val_end]

        valid_indices = range(val_start, val_end)
        fold_oof = [
            {
                'fixtureId': fixture_ids[i],
                'probability': float(valid_pred_proba[j]),
                'trueLabel': int(y_fold_valid[j]),
            }
            for j, i in enumerate(valid_indices)
        ]
        oof_folds.append(fold_oof)

        fold_auc = roc_auc_score(y_fold_valid, valid_pred_proba)
        print(f"Fold {fold+1} ROC AUC: {fold_auc:.4f}")
        fold_auc_scores.append(fold_auc)

        fold_pr_auc = average_precision_score(y_fold_valid, valid_pred_proba)
        print(f"Fold {fold+1} PR AUC: {fold_pr_auc:.4f}")
        fold_pr_auc_scores.append(fold_pr_auc)

        # Feature importance (split/gain from trees)
        fi = booster.feature_importance(importance_type='split')
        feature_importances[f'fold_{fold+1}'] = pd.Series(fi, index=X.columns)

        # Permutation importance using a wrapper that adds zM back
        wrapper = BoosterWithMarketOffset(booster, zM_fold_valid)
        perm = permutation_importance(
            wrapper,
            X_fold_valid,
            y_fold_valid,
            n_repeats=10,
            random_state=42,
            scoring='roc_auc'
        )
        perm_importances[f'fold_{fold+1}'] = pd.Series(
            perm.importances_mean,
            index=X.columns
        )

        # SHAP values (trees explain residuals)
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(X_fold_valid)
        # For binary classification, shap_values is a list; take positive class
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values
        shap_importances[f'fold_{fold+1}'] = pd.Series(
            np.abs(shap_vals).mean(axis=0),
            index=X.columns
        )

    mean_auc = np.mean(fold_auc_scores)
    print(f"Mean ROC AUC across folds: {mean_auc:.4f}")

    mean_pr_auc = np.mean(fold_pr_auc_scores)
    print(f"Mean PR AUC across folds: {mean_pr_auc:.4f}")

    valid_oof_start = fold_size
    oof_logloss = log_loss(oof_true[valid_oof_start:], oof_preds[valid_oof_start:])
    print(f"Log Loss: {oof_logloss:.4f}")

    oof_p_raw = oof_preds[valid_oof_start:]
    oof_y = oof_true[valid_oof_start:]

    A, B, calibrate = fit_platt_calibrator(oof_p_raw, oof_y)
    oof_p_cal = calibrate(oof_p_raw)

    print(f"OOF LogLoss (raw): {log_loss(oof_y, oof_p_raw):.4f}")
    print(f"OOF LogLoss (cal): {log_loss(oof_y, oof_p_cal):.4f}")
    print(f"Platt coeffs: A={A:.4f}, B={B:.4f}")

    # Prepare validation predictions (final)
    # Build a booster on all training data to score X_val if needed
    dtrain_full = lgb.Dataset(X_train, label=y_train, init_score=zM_train)
    dval_full = lgb.Dataset(X_val, label=y_val, init_score=zM_val, reference=dtrain_full)
    booster_full = lgb.train(
        params,
        dtrain_full,
        num_boost_round=num_boost_round,
        valid_sets=[dval_full],
        valid_names=['valid_full'],
        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
    )
    r_hat_val = booster_full.predict(X_val, raw_score=True)
    p_val = sigmoid(zM_val + r_hat_val)
    p_val_cal = calibrate(p_val)

    return {
        'fold_models': fold_models,           # list of boosters (residual learners)
        'booster_full': booster_full,         # trained on full train for val/inference
        'oof_preds': oof_preds[valid_oof_start:],      # final probs (market + residual)
        'oof_preds_cal': oof_p_cal,
        'platt_coeffs': {'A': A, 'B': B},
        'oof_true': oof_true[valid_oof_start:],
        'oof_fixture_ids': oof_fixture_ids[valid_oof_start:],
        'oof_folds': oof_folds,
        'feature_importances': feature_importances,
        'perm_importances': perm_importances,
        'shap_importances': shap_importances,
        'X_val': X_val,
        'y_val': y_val,
        'id_val': id_val,
        'val_probs': p_val,
        'val_probs_cal': p_val_cal,
        'fold_auc_scores': fold_auc_scores,
        'fold_pr_auc_scores': fold_pr_auc_scores,
        'mean_auc': mean_auc,
        'oof_logloss': oof_logloss,
        'zM_val': zM_val,
    }
