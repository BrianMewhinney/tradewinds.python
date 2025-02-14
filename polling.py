import os
import json
import shutil
import time
import pandas as pd
import numpy as np
from xgboost_processing import xgboost_processing
from random_forest.random_forest import random_forest_processing


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


def process_directory(input_dir, output_dir, bestPrecision):
    for subdir in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir)

        if os.path.isdir(subdir_path):
            config_file = os.path.join(subdir_path, 'config.json')
            x_file = os.path.join(subdir_path, 'x.csv')
            y_file = os.path.join(subdir_path, 'y.csv')

            if os.path.isfile(config_file) and os.path.isfile(x_file) and os.path.isfile(y_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)

                #for feature in config['testFeatures']:
                #    print(feature['key'])
                #print(config)
                results = start_test(config, x_file, y_file)
                precision = results['test_precision']
                if precision > bestPrecision:
                    bestPrecision = precision

                print(f"{subdir}:  Processing complete:  Precision: {precision:.6f}  Best Precision: {bestPrecision:.6f}\n")

                serializable_results = make_serializable(results)

                output_subdir_path = os.path.join(output_dir, subdir)
                os.makedirs(output_subdir_path, exist_ok=True)

                results_file = os.path.join(output_subdir_path, 'results.json')
                with open(results_file, 'w') as f:
                    json.dump(serializable_results, f, indent=4)

                out_config_file = os.path.join(output_subdir_path, 'config.json')
                with open(out_config_file, 'w') as f:
                    json.dump(config, f, indent=4)

                in_f_file = os.path.join(subdir_path, 'f.csv')
                out_f_file = os.path.join(output_subdir_path, 'f.csv')
                shutil.copy(in_f_file, out_f_file)
                shutil.rmtree(subdir_path)
    return bestPrecision


def main():
    input_dir = '/Users/brianmewhinney/dev/external/tradewinds-data/input'
    output_dir = '/Users/brianmewhinney/dev/external/tradewinds-data/output'
    bestPrecision = 0
    while True:
        bestPrecision = process_directory(input_dir, output_dir, bestPrecision)
        time.sleep(1)  # Poll every 10 seconds


if __name__ == "__main__":
    main()
