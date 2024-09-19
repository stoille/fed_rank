from os import sendfile
import tensorflow as tf
import tensorflow_federated as tff
from flask import Flask, request, jsonify
import ssl
from shared_model import model_fn
import json
import io

app = Flask(__name__)

# Load candidate articles at server startup
with open('candidate_articles.json', 'r') as f:
    candidate_articles = json.load(f)
    
# Initialize the training process state
state = None
training_process = None

@app.route('/feed', methods=['POST'])
def feed() -> tuple[dict, int]:
    print("initiating feed")
    global state  # Assuming 'state' contains the current global model
    global training_process
    # Build the federated reconstruction training process
    training_process = tff.learning.algorithms.build_fed_recon(
        model_fn=model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
        reconstruction_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1),
        loss_fn=lambda: tf.keras.losses.MeanSquaredError(),
    )
   
    # Initialize the training process state
    state = training_process.initialize()
    print("state initialized")
    # Serialize the global model
    buffer = io.BytesIO()
    tf.saved_model.save(state.model, buffer)
    buffer.seek(0)

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

    return sendfile(
        buffer,
        as_attachment=True,
        download_name='global_model.h5',
        mimetype='application/octet-stream'
    ), 200, {'Content-Type': 'application/json', 'X-Response-Data': json.dumps(response_data)}

# Store user permissions
user_permissions = {}

@app.route('/submit_updates', methods=['POST'])
def submit_updates() -> tuple[dict, int]:
    global state
    global training_process
    print("submitting updates")
    # Receive model updates and permissions from the client
    client_data = request.get_json()
    user_id = client_data['user_id']
    model_updates = client_data['model_updates']
    permissions = client_data['permissions']

    # Store user permissions
    user_permissions[user_id] = permissions

    # Convert model_updates to the expected format (list of tf.Tensors)
    federated_updates = [tf.constant(update, dtype=tf.float32) for update in model_updates]
   
    # Integrate model updates into the federated learning process
    state, metrics = training_process.next(state, [federated_updates])

    # Extract relevant metrics to send back to the client
    client_metrics = {
        'loss': metrics['client_work']['train']['loss'].numpy().item(),
        'accuracy': metrics['client_work']['train']['binary_accuracy'].numpy().item()
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
