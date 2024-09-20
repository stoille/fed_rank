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

app = Flask(__name__)

# Load candidate articles at server startup
with open('candidate_articles.json', 'r') as f:
    candidate_articles = json.load(f)
    
# Initialize the training process state
state = None
training_process = None

# Constants
NUM_USER_FEATURES = int(os.getenv('NUM_USER_FEATURES', '1000'))
NUM_ARTICLES = int(os.getenv('NUM_ARTICLES', '1000'))
EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', '32'))

@app.route('/feed', methods=['POST'])
def feed():
    print("initiating feed")
    global state
    global training_process

    # Build the federated reconstruction training process
    training_process = tff.learning.algorithms.build_fed_recon(
        model_fn=lambda: model_fn(NUM_USER_FEATURES, NUM_ARTICLES, EMBEDDING_DIM)[0],
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
        reconstruction_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1),
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
    model.save('global_model.keras')

    # Load the saved model file into a BytesIO buffer
    with open('global_model.keras', 'rb') as f:
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

@app.route('/submit_updates', methods=['POST'])
def submit_updates() -> tuple[dict, int]:
    global state
    global training_process
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
        
        # Load the model using tff.learning.models.load()
        updated_model = keras.models.load_model(temp_model_path)

    # Extract weights from the updated model
    updated_weights = updated_model.get_weights()

    # Update the global model weights (this is a simplified example)
    # In practice, you'll need to correctly apply the federated averaging logic
    global_model_weights = state.global_model_weights

    # Placeholder for aggregation logic
    # You need to define how to integrate client updates into the server state
    # For example, you might average the weights
    # For illustration purposes:
    new_trainable_weights = []
    for server_w, client_w in zip(global_model_weights.trainable, updated_weights):
        # Take a subset of server weights to match client dimensions
        server_w_subset = tf.slice(server_w, [0] * len(server_w.shape), client_w.shape)
        new_w = server_w_subset + 0.1 * (client_w - server_w_subset)  # Simple aggregation
        # Pad new_w back to original server shape if necessary
        if server_w.shape != new_w.shape:
            padding = [[0, s - n] for s, n in zip(server_w.shape, new_w.shape)]
            new_w = tf.pad(new_w, padding)
        new_trainable_weights.append(new_w)
    
    global_model_weights = tff.learning.models.ModelWeights(
        trainable=new_trainable_weights,
        non_trainable=global_model_weights.non_trainable
    )
    
    # Update the server state with new weights
    state = tff.learning.templates.LearningAlgorithmState(
        global_model_weights=global_model_weights,
        distributor=state.distributor,
        client_work=state.client_work,
        aggregator=state.aggregator,
        finalizer=state.finalizer
    )

    # Extract relevant metrics to send back to the client
    client_metrics = {
        'status': 'Model updates received and processed.'
        # You can add more metrics if needed
    }

    print("sending metrics back to client")
    # Acknowledge receipt of updates and send metrics back to the client
    return jsonify({
        'status': 'success',
        'message': 'Model updates received and processed.',
        'metrics': client_metrics
    }), 200

if __name__ == '__main__':
    # Set up SSL context for HTTPS
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain('cert.pem', 'key.pem')  # Your SSL certificates

    app.run(host='0.0.0.0', port=443, ssl_context=context)
