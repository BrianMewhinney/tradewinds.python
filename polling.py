import os
import json
import shutil
import time
import pandas as pd
import numpy as np
from xgboost_processing import xgboost_processing
from random_forest import random_forest_processing


def start_test(config, x_file, y_file):
    # Placeholder for the external method. Replace with the actual implementation.
    #print("Running test with config:", config)

    # Example of calling your external processing function
    results, importances = random_forest_processing(x_file, y_file)
#    results, importances = xgboost_processing(x_file, y_file)
    return results


def make_serializable(obj):
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.to_dict()  # Convert pandas objects to dictionaries
    elif isinstance(obj, (np.generic, np.ndarray)):
        return obj.tolist()  # Convert numpy objects to lists
    elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
        return obj.item()  # Convert numpy scalars to Python scalars
    elif isinstance(obj, dict):
        return {make_serializable(k): make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(make_serializable(x) for x in obj)
    else:
        return obj  # Return as-is if it's already serializable


def process_directory(input_dir, output_dir, bestAccuracy):
    for subdir in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir)

        if os.path.isdir(subdir_path):
            config_file = os.path.join(subdir_path, 'config.json')
            x_file = os.path.join(subdir_path, 'x.csv')
            y_file = os.path.join(subdir_path, 'y.csv')

            if os.path.isfile(config_file) and os.path.isfile(x_file) and os.path.isfile(y_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                print(config)
                results = start_test(config, x_file, y_file)
                accuracy = results['test_accuracy']
                if accuracy > bestAccuracy:
                    bestAccuracy = accuracy

                print(f"{subdir}:  Processing complete:  Accuracy: {accuracy:.6f}  Best Accuracy: {bestAccuracy:.6f}\n")

                serializable_results = make_serializable(results)

                output_subdir_path = os.path.join(output_dir, subdir)
                os.makedirs(output_subdir_path, exist_ok=True)

                results_file = os.path.join(output_subdir_path, 'results.json')
                with open(results_file, 'w') as f:
                    json.dump(serializable_results, f, indent=4)

                out_config_file = os.path.join(output_subdir_path, 'config.json')
                with open(out_config_file, 'w') as f:
                    json.dump(config, f, indent=4)

                shutil.rmtree(subdir_path)
    return bestAccuracy


def main():
    input_dir = '/Users/brianmewhinney/dev/external/tradewinds-data/input'
    output_dir = '/Users/brianmewhinney/dev/external/tradewinds-data/output'
    bestAccuracy = 0
    while True:
        bestAccuracy = process_directory(input_dir, output_dir, bestAccuracy)
        time.sleep(1)  # Poll every 10 seconds


if __name__ == "__main__":
    main()
