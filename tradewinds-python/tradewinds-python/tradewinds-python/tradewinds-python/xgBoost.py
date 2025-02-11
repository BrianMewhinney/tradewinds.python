import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Load data
X = pd.read_csv('/Users/brianmewhinney/dev/tradewinds-python/data/8/x.csv').values
y = pd.read_csv('/Users/brianmewhinney/dev/tradewinds-python/data/8/y.csv').values.flatten()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_train)

# Grid search for best parameters
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'), 
                           param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_resampled, y_train)

# Evaluate best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(scaler.transform(X_test))
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy:.2f}')

# Feature importance
importances = best_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
print("XGBoost feature importance:")
for idx in sorted_idx:
    print(f"Feature {idx}: {importances[idx]:.4f}")


0.homeOdds
1.homeOdds
2.awayOdds
3.homeSignificantWin-All
4.homeSignificantWin-Ha
5.homeSignificantLoss-All
6.homeSignificantLoss-Ha
7.awaySignificantWin-All
8.awaySignificantWin-Ha
9.awaySignificantLoss-All
10.awaySignificantLoss-Ha
11.homeSignificant-All
12.homeSignificant-Ha
13.awaySignificant-All
14.awaySignificant-Ha
15.homeState-All
16.homeState-Ha
17.awayState-All
18.awayState-Ha
19.homeAttackState-All
20.homeAttackState-Ha
21.awayAttackState-All
22.awayAttackState-Ha
23.homeAttacks-All
24.homeAttacks-Ha
25.awayAttacks-All
26.awayAttacks-Ha
27.homeDangerous-All
28.homeDangerous-Ha
29.awayDangerous-All
