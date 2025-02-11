import os
from random_forest import random_forest_processing
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
        results, importances = random_forest_processing(x_file, y_file)
        all_results.append({
            'directory': subdir,
            'start_time': results['start_time'],
            'execution_time': results['execution_time'],
            'best_params': results['best_params'],
            'best_score': results['best_score'],
            'test_accuracy': results['test_accuracy'],
        })

        accuracy = results['test_accuracy']
        best_score = results['best_score']
        if accuracy > max:
            max = accuracy

        print(f'Test Accuracy: {accuracy:.5f} Best Score: {best_score:5f}  Best Accuracy:{max:.5f}')

        for idx, importance in enumerate(importances):
            feature_importance_aggregate[idx] += importance
    else:
        print(f"Skipping directory (files not found): {subdir}")

sorted_results = sorted(all_results, key=lambda x: x['test_accuracy'], reverse=True)
summary_file_path = os.path.join(base_directory, 'summary_results.txt')

print(sorted_results[0]['best_params'])

with open(summary_file_path, 'w') as f:
    for result in sorted_results:
        f.write(f"Directory: {result['directory']}\n")
        f.write(f"Start Time: {result['start_time']}\n")
        f.write(f"Execution Time: {result['execution_time']:.2f} seconds\n")
        f.write(f"Best Parameters: {result['best_params']}\n")
        f.write(f"Best Score: {result['best_score']:.6f}\n")
        f.write(f"Test Accuracy: {result['test_accuracy']:.5f}\n")
#         f.write("Feature Importance:\n")
#          for feature, (idx, importance) in result['feature_importance'].items():
#              f.write(f"  Feature {idx}: {importance:.4f}\n")
#          f.write("Permutation Feature Importance:\n")
#          for idx, importance in result['permutation_importance'].items():
#              f.write(f"  Feature {idx}: {importance:.4f}\n")

        f.write("\n")

    f.write("Aggregated Feature Importances:\n")
    for idx, total_importance in feature_importance_aggregate.items():
        f.write(f"Feature {idx}: {total_importance:.4f}\n")

