from os import sendfile
import tensorflow as tf
import tensorflow_federated as tff
from flask import Flask, request, jsonify, send_file
import ssl
from shared_model import model_fn
import json
import io
import os
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow import keras
import tempfile
import numpy as np
import platform

# global paths
CERT_PATH = os.path.join('data', 'cert.pem')
KEY_PATH = os.path.join('data', 'key.pem')
CANDIDATE_ARTICLES_PATH = os.path.join('data', 'candidate_articles.json')
GLOBAL_MODEL_PATH = os.path.join('data', 'global_model.keras')

app = Flask(__name__)

# Load candidate articles at server startup
with open(CANDIDATE_ARTICLES_PATH, 'r') as f:
    candidate_articles = json.load(f)
    
# Initialize the training process state
state = None
training_process = None

# Constants
NUM_USER_FEATURES = int(os.getenv('NUM_USER_FEATURES', '1000'))
NUM_ARTICLES = int(os.getenv('NUM_ARTICLES', '1000'))
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
        optimizer = tf.keras.optimizers.legacy.SGD
    else:
        optimizer = tf.keras.optimizers.SGD

    # Build the federated reconstruction training process
    training_process = tff.learning.algorithms.build_fed_recon(
        model_fn=lambda: model_fn(NUM_USER_FEATURES, NUM_ARTICLES, EMBEDDING_DIM)[0],
        client_optimizer_fn=lambda: optimizer(learning_rate=0.1),
        server_optimizer_fn=lambda: optimizer(learning_rate=1.0),
        reconstruction_optimizer_fn=lambda: optimizer(learning_rate=0.1),
        loss_fn=lambda: tf.keras.losses.BinaryCrossentropy(from_logits=True),
    )
    
    # Set the local Python execution context
    execution_contexts.set_sync_local_cpp_execution_context()

    # Initialize the training process state
    state = training_process.initialize()
    print("state initialized")

    # Extract global model weights from the state
    global_model_weights = state.global_model_weights

    # Reconstruct the Keras model using model_fn
    model = model_fn(NUM_USER_FEATURES, NUM_ARTICLES, EMBEDDING_DIM)[1]

    # Assign the global model weights to the Keras model
    model_weights = tff.learning.models.ModelWeights(
        trainable=global_model_weights.trainable,
        non_trainable=global_model_weights.non_trainable
    )
    model_weights.assign_weights_to(model)

    # Save the model to a file
    model.save(GLOBAL_MODEL_PATH)

    # Load the saved model file into a BytesIO buffer
    with open(GLOBAL_MODEL_PATH, 'rb') as f:
        model_buffer = io.BytesIO(f.read())

    # Reset the buffer's position to the beginning
    model_buffer.seek(0)

    response_data = {
        'message': 'Global model and candidate articles provided. Use POST /submit_updates to provide model updates.',
        'candidate_articles': candidate_articles,
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
    user_ids = np.random.randint(0, NUM_USER_FEATURES, size=(num_samples, 1))
    article_ids = np.random.randint(0, NUM_ARTICLES, size=(num_samples, 1))
    labels = np.random.randint(0, 2, size=(num_samples, 1))  # 1 if the user interacted with the article, else 0

    dataset = tf.data.Dataset.from_tensor_slices((
        {'user_input': user_ids, 'article_input': article_ids},
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
    all_article_ids = []

    for batch in validation_dataset:
        inputs, labels = batch
        predictions = model.predict(inputs, verbose=0).flatten()
        all_labels.extend(labels.numpy().flatten())
        all_predictions.extend(predictions)
        all_user_ids.extend(inputs['user_input'].numpy().flatten())
        all_article_ids.extend(inputs['article_input'].numpy().flatten())

    # Organize data by user
    user_data = {}
    for user_id, article_id, label, prediction in zip(all_user_ids, all_article_ids, all_labels, all_predictions):
        if user_id not in user_data:
            user_data[user_id] = {'labels': {}, 'predictions': {}}
        user_data[user_id]['labels'][article_id] = label
        user_data[user_id]['predictions'][article_id] = prediction

    # Compute metrics
    precision_at_k = []
    recall_at_k = []
    ndcg_at_k = []

    for user_id, data in user_data.items():
        labels = data['labels']
        predictions = data['predictions']

        # Get the list of articles sorted by predicted score
        ranked_articles = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        top_k_articles = [article_id for article_id, _ in ranked_articles[:TOP_K]]

        # Relevant articles are those with label == 1
        relevant_articles = [article_id for article_id, label in labels.items() if label == 1]

        # Compute Precision@K
        num_relevant_in_top_k = sum([1 for article_id in top_k_articles if labels.get(article_id, 0) == 1])
        precision = num_relevant_in_top_k / TOP_K if TOP_K > 0 else 0
        precision_at_k.append(precision)

        # Compute Recall@K
        num_relevant = len(relevant_articles)
        recall = num_relevant_in_top_k / num_relevant if num_relevant > 0 else 0
        recall_at_k.append(recall)

        # Compute NDCG@K
        dcg = 0.0
        idcg = 0.0
        for i, article_id in enumerate(top_k_articles):
            rel = labels.get(article_id, 0)
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
def submit_updates() -> tuple[dict, int]:
    global state
    global training_process
    global client_updates_list
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
        updated_model = keras.models.load_model(temp_model_path)

    # Extract weights from the updated model
    client_weights = updated_model.get_weights()

    # Append the client weights to the list
    client_updates_list.append(client_weights)

    # Initialize default metrics
    global_model_loss = None
    global_model_accuracy = None
    recommender_metrics = {}
    ranked_results = []

    # Check if enough client updates have been collected to perform aggregation
    if len(client_updates_list) >= NUM_CLIENTS_PER_ROUND:
        print("Aggregating client updates...")
        # Initialize a list to hold the aggregated weights
        aggregated_weights = []

        # Number of layers in the model
        num_layers = len(client_updates_list[0])

        # Perform federated averaging
        for layer_idx in range(num_layers):
            # Collect the weights for the current layer from all clients
            layer_weights = [client_weights[layer_idx] for client_weights in client_updates_list]
            # Convert the list to a numpy array for averaging
            layer_weights = np.array(layer_weights)
            # Compute the mean of the weights
            mean_weights = np.mean(layer_weights, axis=0)
            # Append the averaged weights to the aggregated_weights list
            aggregated_weights.append(mean_weights)
        
        # Update the global model weights
        # Reconstruct the Keras model using model_fn
        model = model_fn(NUM_USER_FEATURES, NUM_ARTICLES, EMBEDDING_DIM)[1]
        # Assign the aggregated weights to the model
        model.set_weights(aggregated_weights)

        # Compile the model with SGD optimizer and BinaryCrossentropy loss
        # Check if running on M1/M2 Mac
        is_m1_mac = platform.processor() == 'arm'
        
        if is_m1_mac:
            optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.1)
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
        
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        model.compile(optimizer, loss, metrics=['accuracy'])
        
        # Evaluate the updated global model on a validation dataset
        validation_dataset = load_validation_data()
        evaluation_results = model.evaluate(validation_dataset, verbose=0)
        global_model_loss, global_model_accuracy = evaluation_results

        # Compute recommender system metrics
        recommender_metrics = compute_recommender_metrics(model, validation_dataset)

        # Rank articles
        ranked_results = rank_articles(model, candidate_articles)

        # Extract global model weights from the updated model
        global_model_weights = tff.learning.models.ModelWeights.from_model(model)
        
        # Update the server state with new weights
        state = tff.learning.templates.LearningAlgorithmState(
            global_model_weights=global_model_weights,
            distributor=state.distributor,
            client_work=state.client_work,
            aggregator=state.aggregator,
            finalizer=state.finalizer
        ) 
        
        # Clear the client updates list for the next round
        client_updates_list = []
        print("Global model updated.")

    # Extract relevant metrics to send back to the client
    client_metrics = {
        'status': 'Model updates received and processed.',
        'global_model_loss': global_model_loss,
        'global_model_accuracy': global_model_accuracy,
        'global_model_precision_at_k': recommender_metrics.get('precision_at_k'),
        'global_model_recall_at_k': recommender_metrics.get('recall_at_k'),
        'global_model_ndcg_at_k': recommender_metrics.get('ndcg_at_k'),
        'ranked_results': ranked_results[:10]  # Send top 10 ranked articles
    }

    print("sending metrics back to client")
    # Acknowledge receipt of updates and send metrics back to the client
    return jsonify({
        'status': 'success',
        'message': 'Model updates received and processed.',
        'metrics': client_metrics
    }), 200

def rank_articles(model, candidate_articles):
    article_id_to_index = {
        article['id']: idx for idx, article in enumerate(candidate_articles)
    }
    test_article_indices = [
        article_id_to_index[article['id']] for article in candidate_articles
    ]
    test_article_lengths = [len(article['headline']) for article in candidate_articles]
    
    test_article_indices_tensor = tf.constant(test_article_indices, dtype=tf.int32)
    test_article_lengths_tensor = tf.constant(test_article_lengths, dtype=tf.float32)
    
    predictions = model.predict(
        {'article_input': test_article_indices_tensor, 'user_input': np.zeros((len(candidate_articles), 1))}
    )

    ranked_results = sorted(
        zip(
            [article['id'] for article in candidate_articles],
            predictions.flatten().astype(float)
        ),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked_results

if __name__ == '__main__':
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(CERT_PATH, KEY_PATH)  # Your SSL certificates

    app.run(host='0.0.0.0', port=443, ssl_context=context)
