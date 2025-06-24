import pika
import json
from random_forest.random_forest2 import random_forest_processing
from light_gbm.light_gbm import light_gbm_predictor
import pandas as pd
import numpy as np
from evaluate_model import evaluate_model
from model_prediction import model_prediction
from sklearn.metrics import f1_score

#CONNECTION_HOST = '192.168.1.53'
CONNECTION_HOST = '192.168.4.77'

connection = pika.BlockingConnection(
    pika.ConnectionParameters(
        host=CONNECTION_HOST,
        port=5672,
        credentials=pika.PlainCredentials('tradewinds', 'tradewinds')
    )
)
executionChannel = connection.channel()
executionChannel.queue_declare(queue='model_executions', durable=True)
executionChannel.basic_qos(prefetch_count=1)
resultsChannel = connection.channel()
resultsChannel.queue_declare(queue='execution_results', durable=True)


def find_best_threshold(y_true, y_proba):
    """Find the best threshold for F1 using unique probabilities."""
    candidate_thresholds = np.unique(y_proba)
    best_thresh = 0.5
    best_f1 = 0
    best_pred = None

    for t in candidate_thresholds:
        pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_pred = pred.copy()

    print(f"Best threshold: {best_thresh:.4f}, Best F1: {best_f1:.4f}")
    return best_thresh, best_f1, best_pred, y_proba

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



def send_to_result_queue(data):
   message = json.dumps(data)
   resultsChannel.basic_publish(exchange='',
                         routing_key='execution_results',
                         body=message,
                         properties=pika.BasicProperties(
                             delivery_mode=2,  # Make message persistent
                         ))

   print("Message has been sent")


def callback(ch, method, properties, body):
    #'''
    data = json.loads(body)
    config = data['config']
    print(f"Received execution request for simulation: {config['simulationId']}  execution: {config['executionId']}  league {config['leagueId']}")

    # train the lightgbm model on the data passed in
    results = light_gbm_predictor(data['x'], data['y'], data['predX'])
    oof_preds = results['oof_preds']
    oof_true = results['oof_true']
    oof_fixture_ids = results['oof_fixture_ids']

    #oof_preds, oof_true = results['oof_preds'], results['oof_true']
    # Find best threshold on OOF predictions
    best_thresh, best_f1, oof_pred_labels, oof_probas_used = find_best_threshold(oof_true, oof_preds)

    # Capture the results of the testing set on the trained model
    test_metrics, y_pred_np, y_proba_np = evaluate_model(results['fold_models'], results['X_val'], results['y_val'], best_thresh)
    test_metrics['oof_preds'] = oof_preds
    test_metrics['oof_true'] = oof_true
    test_metrics['oof_fixture_ids'] = oof_fixture_ids
    test_metrics['mean_auc'] = results['mean_auc']
    test_metrics['fold_auc_scores'] = results['fold_auc_scores']
    test_metrics['fold_pr_auc_scores'] = results['fold_pr_auc_scores']


    metrics = make_serializable(test_metrics)
    y_pred = make_serializable(y_pred_np)
    y_proba = make_serializable(y_proba_np)
    y_val = make_serializable(results['y_val'])
    id_val = make_serializable(results['id_val'])
    feature_importances = make_serializable(results['feature_importances'])
    #shap_summary = make_serializable(results['shap_summary_df'])
    permutation_importance = make_serializable(results['perm_importances'])
    send_to_result_queue((metrics, config, y_pred, y_val, id_val, feature_importances, permutation_importance, y_proba))
    #shap_values = make_serializable(shap_values_np)

    #'''

    ch.basic_ack(delivery_tag=method.delivery_tag)


executionChannel.basic_consume(queue='model_executions', on_message_callback=callback)

print(' [*] Waiting for messages. To exit press CTRL+C')
executionChannel.start_consuming()
