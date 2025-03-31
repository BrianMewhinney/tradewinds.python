import pika
import json
from random_forest.random_forest import random_forest_processing
import pandas as pd
import numpy as np


connection = pika.BlockingConnection(
    pika.ConnectionParameters(
        host='192.168.4.64',
        port=5672,
        credentials=pika.PlainCredentials('tradewinds', 'tradewinds')
    )
)
executionChannel = connection.channel()
executionChannel.queue_declare(queue='prod_executions', durable=True)
executionChannel.basic_qos(prefetch_count=1)
resultsChannel = connection.channel()
resultsChannel.queue_declare(queue='execution_results', durable=True)

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
    data = json.loads(body)
    config = data['config']
    print(f"Received execution request for simulation: {config['simulationId']}  execution: {config['executionId']}  league {config['leagueId']}")

    y_pred, y_test, x_test, npResults, importances = random_forest_processing(data['x'], data['y'])
    print("Message processing complete")
    results = make_serializable(npResults)
    send_to_result_queue((results, config, y_pred, y_test, x_test, data['f']))
    ch.basic_ack(delivery_tag=method.delivery_tag)


executionChannel.basic_consume(queue='prod_executions', on_message_callback=callback)

print(' [*] Waiting for messages. To exit press CTRL+C')
executionChannel.start_consuming()
