import os
from xgboost_processing import xgboost_processing

from collections import defaultdict

# Base directory containing the subdirectories
base_directory = '/Users/brianmewhinney/dev/external/tradewinds-data/simulation/8'
feature_importance_aggregate = defaultdict(float)  # Aggregate feature importances

all_results = []
cnt = 1
max = 0
for subdir, _, _ in os.walk(base_directory):
    x_file = os.path.join(subdir, 'x.csv')
    y_file = os.path.join(subdir, 'y.csv')

    if os.path.isfile(x_file) and os.path.isfile(y_file):
        print(f"\nProcessing directory: {subdir}.  Pass: {cnt}")
        cnt = cnt + 1
        xgboost_processing(x_file, y_file)
    else:
        print(f"Skipping directory (files not found): {subdir}")


