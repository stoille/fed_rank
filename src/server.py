from os import sendfile
import tensorflow as tf
import tensorflow_federated as tff
from flask import Flask, request, jsonify, send_file
import ssl
from shared_model import model_fn, get_keras_model
import json
import io
import os
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow import keras
import tempfile
import numpy as np
import platform
from tensorflow_federated.python.core.impl.types import computation_types

# global paths
CERT_PATH = os.path.join('data', 'cert.pem')
KEY_PATH = os.path.join('data', 'key.pem')
CANDIDATE_ARTICLES_PATH = os.path.join('data', 'candidate_items.json')
GLOBAL_MODEL_PATH = os.path.join('data', 'global_model.keras')

app = Flask(__name__)

# Load candidate articles at server startup
with open(CANDIDATE_ARTICLES_PATH, 'r') as f:
    candidate_items = json.load(f)
    
# Initialize the training process state
state = None
training_process = None
client_datasets = []

# Constants
NUM_USERS = int(os.getenv('NUM_USERS', '1000'))
NUM_ITEMS = int(os.getenv('NUM_ITEMS', '1000'))
EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', '32'))
TOP_K = 10  # Define K for Precision@K and Recall@K

# Global variables to store client updates
client_updates_list = []
NUM_CLIENTS_PER_ROUND = 3  # Adjust this number based on your requirements

@app.route('/feed', methods=['POST'])
def feed():
    print("initiating feed")
    global state
    global training_process

    # Check if running on M1/M2 Mac
    is_m1_mac = platform.processor() == 'arm'
    
    if is_m1_mac:
        optimizer = tf.keras.optimizers.legacy.Adam
    else:
        optimizer = tf.keras.optimizers.Adam

    # Build the federated learning process
    training_process = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=lambda: model_fn()[0],  # Use the updated model_fn
        client_optimizer_fn=lambda: optimizer(learning_rate=0.1),
        server_optimizer_fn=lambda: optimizer(learning_rate=1.0),
    )
    
    # Set the local Python execution context. This needs to be set to run TFF locally
    execution_contexts.set_sync_local_cpp_execution_context()

    # Initialize the training process state
    state = training_process.initialize()

    # Reconstruct the Keras model to save and send to clients
    keras_model = get_keras_model()

    # Save the model to a file
    keras_model.save(GLOBAL_MODEL_PATH)

    # Load the saved model file into a BytesIO buffer
    with open(GLOBAL_MODEL_PATH, 'rb') as f:
        model_buffer = io.BytesIO(f.read())

    # Reset the buffer's position to the beginning
    model_buffer.seek(0)

    response_data = {
        'message': 'Global model and candidate articles provided. Use POST /submit_updates to provide model updates.',
        'candidate_items': candidate_items,
        'permissions_requested': {
            'essential': {
                'enabled': True,
                'description': 'Necessary for the functioning of the federated learning system.',
                'data_usage': ['model_updates_aggregation']
            },
            'functional': {
                'enabled': True,
                'description': 'Enables enhanced features and personalization.',
                'data_usage': ['user_preferences_reconstruction']
            },
            'analytics': {
                'enabled': False,
                'description': 'Helps us understand how our system is used and improve it. You get enhanced personalization.',
                'data_usage': ['usage_statistics', 'performance_metrics']
            },
            'third_party': {
                'enabled': False,
                'description': 'Allows sharing of data with trusted partners for research purposes. You get enhanced personalization.',
                'data_usage': ['collaborative_research']
            }
        },
        'data_usage_terms': """
        1. Essential data is used solely for model aggregation and system functionality.
        2. Functional data improves user experience through personalization.
        3. Analytics data, if enabled, helps improve our services.
        4. No data will be shared with third parties unless explicitly enabled.
        5. All data processing complies with GDPR and applicable data protection laws.
        """,
        'message': 'next POST /submit_updates to provide model updates and permission settings.'
    }

    # Attach the serialized model as a file in the response
    return send_file(
        model_buffer,
        as_attachment=True,
        download_name='global_model.keras',
        mimetype='application/octet-stream'
    ), 200, {'X-Response-Data': json.dumps(response_data)}

# Store user permissions
user_permissions = {}

def load_validation_data():
    """
    Load or generate validation data that includes user interactions.
    """
    # Example validation data with user IDs, article IDs, and labels indicating interactions
    num_samples = 1000
    user_ids = np.random.randint(0, NUM_USERS, size=(num_samples, 1))
    item_ids = np.random.randint(0, NUM_ITEMS, size=(num_samples, 1))
    labels = np.random.randint(0, 2, size=(num_samples, 1))  # 1 if the user interacted with the article, else 0

    dataset = tf.data.Dataset.from_tensor_slices((
        {'user_input': user_ids, 'item_input': item_ids},
        labels
    ))
    dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    return dataset

def compute_recommender_metrics(model, validation_dataset):
    """
    Compute recommender system metrics.
    """
    all_labels = []
    all_predictions = []
    all_user_ids = []
    all_item_ids = []

    for batch in validation_dataset:
        inputs, labels = batch
        predictions = model.predict(inputs, verbose=0).flatten()
        all_labels.extend(labels.numpy().flatten())
        all_predictions.extend(predictions)
        all_user_ids.extend(inputs['user_input'].numpy().flatten())
        all_item_ids.extend(inputs['item_input'].numpy().flatten())

    # Organize data by user
    user_data = {}
    for user_id, item_id, label, prediction in zip(all_user_ids, all_item_ids, all_labels, all_predictions):
        if user_id not in user_data:
            user_data[user_id] = {'labels': {}, 'predictions': {}}
        user_data[user_id]['labels'][item_id] = label
        user_data[user_id]['predictions'][item_id] = prediction

    # Compute metrics
    precision_at_k = []
    recall_at_k = []
    ndcg_at_k = []

    for user_id, data in user_data.items():
        labels = data['labels']
        predictions = data['predictions']

        # Get the list of articles sorted by predicted score
        ranked_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        top_k_items = [item_id for item_id, _ in ranked_items[:TOP_K]]

        # Relevant articles are those with label == 1
        relevant_items = [item_id for item_id, label in labels.items() if label == 1]

        # Compute Precision@K
        num_relevant_in_top_k = sum([1 for item_id in top_k_items if labels.get(item_id, 0) == 1])
        precision = num_relevant_in_top_k / TOP_K if TOP_K > 0 else 0
        precision_at_k.append(precision)

        # Compute Recall@K
        num_relevant = len(relevant_items)
        recall = num_relevant_in_top_k / num_relevant if num_relevant > 0 else 0
        recall_at_k.append(recall)

        # Compute NDCG@K
        dcg = 0.0
        idcg = 0.0
        for i, item_id in enumerate(top_k_items):
            rel = labels.get(item_id, 0)
            dcg += (2 ** rel - 1) / np.log2(i + 2)  # i + 2 because index starts at 0

        # Ideal DCG
        sorted_relevances = sorted([label for label in labels.values()], reverse=True)
        for i, rel in enumerate(sorted_relevances[:TOP_K]):
            idcg += (2 ** rel - 1) / np.log2(i + 2)

        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_at_k.append(ndcg)

    # Aggregate metrics
    metrics = {
        'precision_at_k': np.mean(precision_at_k),
        'recall_at_k': np.mean(recall_at_k),
        'ndcg_at_k': np.mean(ndcg_at_k)
    }
    return metrics

@app.route('/submit_updates', methods=['POST'])
def submit_updates():
    try:
        global state
        global training_process
        global client_datasets
        print("submitting updates")
        
        # Receive model updates and permissions from the client
        user_id = request.form['user_id']
        permissions = json.loads(request.form['permissions'])
        
        # Store user permissions
        user_permissions[user_id] = permissions

        # Save the uploaded file to a temporary directory
        model_file = request.files['model']
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_model_path = os.path.join(temp_dir, 'local_model.keras')
            model_file.save(temp_model_path)
            
            # Load the model using keras.models.load_model()
            server_model = keras.models.load_model(temp_model_path, compile=False)

        # Convert the updated model to a TFF dataset
        client_dataset = tf_model_to_dataset(server_model)
        client_datasets.append(client_dataset)

        # Initialize default metrics
        global_model_loss = None
        global_model_accuracy = None
        recommender_metrics = {}

        # Check if enough client updates have been collected to perform aggregation
        if len(client_datasets) >= NUM_CLIENTS_PER_ROUND:
            print("Aggregating client updates...")
            
            # Perform federated learning round
            state, metrics = training_process.next(state, client_datasets)

            # Extract metrics
            global_model_loss = metrics['loss']
            global_model_accuracy = metrics['accuracy']

            # Reconstruct the Keras model using model_fn
            _, server_model = model_fn()
            
            # Assign the updated weights to the model
            tff.learning.models.ModelWeights.assign_weights_to(state.global_model_weights, server_model)

            # Compute recommender system metrics
            validation_dataset = load_validation_data()
            recommender_metrics = compute_recommender_metrics(server_model, validation_dataset)
            
            # Clear the client datasets list for the next round
            client_datasets = []
            
            # Save the model to a file
            server_model.save(GLOBAL_MODEL_PATH)
            print("Global model updated.")

        # Extract relevant metrics to send back to the client
        client_metrics = {
            'status': 'Model updates received and processed.',
            'global_model_loss': global_model_loss,
            'global_model_accuracy': global_model_accuracy,
            'global_model_precision_at_k': recommender_metrics.get('precision_at_k'),
            'global_model_recall_at_k': recommender_metrics.get('recall_at_k'),
            'global_model_ndcg_at_k': recommender_metrics.get('ndcg_at_k'),
        }

        # Serialize the response data, ensuring no newlines
        response_json = json.dumps({
            'status': 'success',
            'message': 'Model updates received and processed.',
            'metrics': client_metrics
        }).replace('\n', '')

        # Load the saved model file into a BytesIO buffer
        with open(GLOBAL_MODEL_PATH, 'rb') as f:
            model_buffer = io.BytesIO(f.read())

        # Reset the buffer's position to the beginning
        model_buffer.seek(0)

        return send_file(
            model_buffer,
            as_attachment=True,
            download_name='global_model.keras',
            mimetype='application/octet-stream'
        ), 200, {'X-Response-Data': response_json}

    except Exception as e:
        app.logger.error(f"Error in submit_updates: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

def tf_model_to_dataset(model):
    """Convert a TensorFlow model to a TFF dataset."""
    weights = model.get_weights()
    dataset = tf.data.Dataset.from_tensor_slices(weights)
    return dataset.batch(1)

if __name__ == '__main__':
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(CERT_PATH, KEY_PATH)  # Your SSL certificates

    app.run(host='0.0.0.0', port=443, ssl_context=context)
