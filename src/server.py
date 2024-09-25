import io
import os
import ssl
import json
import tempfile
import platform
import functools
import collections
import numpy as np
import pandas as pd
from os import sendfile
import tensorflow as tf
from tensorflow import keras
import tensorflow_federated as tff
from typing import Optional, List, Tuple
from flask import Flask, request, jsonify, send_file
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.backends.native import execution_contexts
import pprint

# global paths
CERT_PATH = os.path.join('data', 'cert.pem')
KEY_PATH = os.path.join('data', 'key.pem')
ARTICLES_PATH = os.path.join('data', 'articles.csv')
GLOBAL_MODEL_PATH = os.path.join('data', 'global_model.keras')
# Constants
NUM_EPOCHS = int(os.environ.get('NUM_EPOCHS', 100))
PATIENCE = int(os.environ.get('PATIENCE', 3))
MOMENTUM = float(os.environ.get('MOMENTUM', 0.96))
LEARNING_RATE = float(os.environ.get('LEARNING_RATE', 0.005))

# Constants
NUM_USERS = int(os.getenv('NUM_USERS', '100'))
NUM_ITEMS = int(os.getenv('NUM_ITEMS', '100'))
EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', '32'))
MAX_EXAMPLES_PER_USER=300,
MAX_CLIENTS=2000
TOP_K = 10  # Define K for Precision@K and Recall@K
RANDOM_SEED = 42

# Global variables to store client updates
NUM_CLIENTS_PER_ROUND = 3  # Adjust this number based on your requirements
MAX_EXAMPLES_PER_USER = 100

app = Flask(__name__)

@app.route('/feed', methods=['POST'])
def feed():
    print("initiating feed")
    global state
    global global_model
    global training_state
    global candidate_items
    
    # Load the ratings data
    articles_df, ratings_df, tf_train_datasets, tf_val_datasets = load_articles_and_ratings_data()  
    
    # Train the model
    training_state, eval_state, train_metrics = train_model(tf_train_datasets, tf_val_datasets)
     
    # Get keras model from TFF state
    keras_model = get_keras_model()
    
    # Set the weights of the keras model
    training_model_weights = training_process.get_model_weights(training_state)
    # Correctly assign weights to the Keras model
    tff.learning.models.ModelWeights.assign_weights_to(training_model_weights, keras_model)

    # Save the model to a file
    keras_model.save(GLOBAL_MODEL_PATH)

    # Load the saved model file into a BytesIO buffer
    with open(GLOBAL_MODEL_PATH, 'rb') as f:
        model_buffer = io.BytesIO(f.read())

    # Reset the buffer's position to the beginning
    model_buffer.seek(0)
    
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_serializable(v) for v in obj)
        else:
            return obj

    response_data = {
        'message': 'Global model and candidate articles provided. Use POST /submit_updates to provide model updates.',
        'candidate_items': articles_df.to_dict(orient='records'),
        'permissions_requested': {
            'essential': {
                'description': 'Necessary for the functioning of the federated learning system.'
            },
            'functional': {
                'description': 'Enables enhanced features and personalization.'
            },
            'analytics': {
                'description': 'Helps us understand how our system is used and improve it. You get enhanced personalization.'
            },
            'third_party': {
                'description': 'Allows sharing of data with trusted partners for research purposes. You get enhanced personalization.'
            }
        },
        'metrics': {k: convert_to_serializable(v) for k, v in train_metrics.items()},
        'message': 'next POST /submit_updates to provide model updates and permission settings.'
    }

    # Attach the serialized model as a file in the response
    return send_file(
        model_buffer,
        as_attachment=True,
        download_name='server_model.keras',
        mimetype='application/octet-stream'
    ), 200, {'X-Response-Data': json.dumps(response_data)}

client_datasets = []

@app.route('/submit_updates', methods=['POST'])
def submit_updates():
    global training_state
    global client_model_weights_list

    execution_contexts.set_sync_local_cpp_execution_context()
    try:
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
            
            # Load the model with custom_objects if needed
            local_model = keras.models.load_model(
                temp_model_path,
                compile=False,
                custom_objects={'UserEmbedding': UserEmbedding}
            )
            
            # Extract the model weights
            client_weights = local_model.get_weights()
            # Append the client weights to the list
            client_model_weights_list.append(client_weights)

        # Check if enough client updates have been collected to perform aggregation
        if len(client_model_weights_list) >= NUM_CLIENTS_PER_ROUND:
            print("Aggregating client updates...")

            # Perform Federated Averaging
            # Average the weights across clients
            num_clients = len(client_model_weights_list)

            aggregated_weights = []
            for weights in zip(*client_model_weights_list):
                aggregated_weights.append(
                    np.mean(weights, axis=0)
                )

            # Update the global model with the aggregated weights
            global_model = get_keras_model()
            global_model.set_weights(aggregated_weights)

            # Update the training_state with the new global model weights
            new_global_model_weights = tff.learning.models.ModelWeights.from_model(global_model)
            training_state = tff.learning.templates.ServerState(
                model=new_global_model_weights,
                optimizer_state=training_state.optimizer_state,
                delta_aggregate_state=training_state.delta_aggregate_state,
                model_broadcast_state=training_state.model_broadcast_state
            )

            # Save the updated global model
            global_model.save(GLOBAL_MODEL_PATH)
            print("Global model updated.")

            # Clear the client model weights list for the next round
            client_model_weights_list = []

        # Send response
        return jsonify({'status': 'success', 'message': 'Model updates received and processed.'}), 200

    except Exception as e:
        app.logger.error(f"Error in submit_updates: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
    

def get_keras_model():
    global NUM_ITEMS
    global NUM_USERS
    global EMBEDDING_DIM
    
    item_input = tf.keras.Input(shape=(1,), dtype=tf.int32, name='item_input')
    user_input = tf.keras.Input(shape=(1,), dtype=tf.int32, name='user_input')

    item_embedding_layer = tf.keras.layers.Embedding(
        input_dim=NUM_ITEMS,
        output_dim=EMBEDDING_DIM,
        embeddings_initializer='he_normal'
    )
    user_embedding_layer = tf.keras.layers.Embedding(
        input_dim=NUM_USERS,
        output_dim=EMBEDDING_DIM,
        embeddings_initializer='he_normal'
    )

    item_embedding = item_embedding_layer(item_input)
    user_embedding = user_embedding_layer(user_input)
    
    concatenated = tf.keras.layers.Concatenate()([user_embedding, item_embedding])
    dense1 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal')(concatenated)
    bn1 = tf.keras.layers.BatchNormalization()(dense1)
    dense2 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal')(bn1)
    bn2 = tf.keras.layers.BatchNormalization()(dense2)
    output = tf.keras.layers.Dense(1, activation='linear', kernel_initializer='he_normal')(bn2)
    
    model = tf.keras.Model(inputs=[item_input, user_input], outputs=output)
    
    # Assign weights to the model
    tff.learning.models.ModelWeights.assign_weights_to(eval_state.global_model_weights, model)
    
    return model

def load_articles_and_ratings_data(
    data_directory: str = "data"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[tf.data.Dataset], List[tf.data.Dataset]]:
    """Loads pandas DataFrames for ratings, articles, users from data directory."""
    
    # Define the data types for each column
    dtype_ratings = {
        'UserID': int,
        'ItemID': int,
        'Rating': float,
        'Timestamp': str
    }
    
    # Load ratings data
    ratings_df = pd.read_csv(
        os.path.join(data_directory, "ratings.csv"),
        sep="::",
        dtype=dtype_ratings,
        names=["UserID", "ItemID", "Rating", "Timestamp"], 
        engine="python")
    
    # Create TensorFlow datasets
    tf_datasets = create_tf_datasets(ratings_df, batch_size=1)
    
    # Split the datasets into training and validation
    tf_train_datasets, tf_val_datasets = split_train_val_datasets(tf_datasets)

    # Load articles data
    articles_df = pd.read_csv(
        os.path.join(data_directory, "articles.csv"),
        sep="::",
        names=["ItemId","Headline","Subheadline","Author","PublishDate","Category","ThumbnailUrl"], 
        engine="python", 
        encoding="ISO-8859-1")

    return articles_df, ratings_df, tf_train_datasets, tf_val_datasets

def create_tf_datasets(ratings_df: pd.DataFrame, batch_size: int = 1) -> List[tf.data.Dataset]:
    def rating_batch_map_fn(rating_batch):
        return collections.OrderedDict([
            ("x", tf.cast(rating_batch[:, 1:2] - 1, tf.int64)),  # ItemID (adjusted for 0-based indexing)
            ("y", tf.cast((rating_batch[:, 2:3]) / 4, tf.float32))  # Normalized Rating
        ])
    
    tf_datasets = []
    for user_id in ratings_df['UserID'].unique():
        user_ratings = ratings_df[ratings_df['UserID'] == user_id][['UserID', 'ItemID', 'Rating']].values
        tf_dataset = tf.data.Dataset.from_tensor_slices(user_ratings)
        tf_dataset = tf_dataset.batch(batch_size).map(rating_batch_map_fn)
        tf_datasets.append(tf_dataset)
    
    return tf_datasets

def split_train_val_datasets(tf_datasets: List[tf.data.Dataset], 
                             train_ratio: float = 0.8) -> Tuple[List[tf.data.Dataset], List[tf.data.Dataset]]:
    tf_train_datasets = []
    tf_val_datasets = []
    
    for dataset in tf_datasets:
        dataset_size = tf.data.experimental.cardinality(dataset).numpy()
        if dataset_size == 0:
            continue
        train_size = int(train_ratio * dataset_size)
        
        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size)
        
        if tf.data.experimental.cardinality(train_dataset).numpy() > 0:
            tf_train_datasets.append(train_dataset)
        if tf.data.experimental.cardinality(val_dataset).numpy() > 0:
            tf_val_datasets.append(val_dataset)
    
    return tf_train_datasets, tf_val_datasets

class UserEmbedding(tf.keras.layers.Layer):
    """Keras layer representing an embedding for a single user, used below."""

    def __init__(self, num_latent_factors, **kwargs):
        super().__init__(**kwargs)
        self.num_latent_factors = num_latent_factors

    def build(self, input_shape):
        self.embedding = self.add_weight(
            shape=(1, self.num_latent_factors),
            initializer='uniform',
            dtype=tf.float32,
            name='UserEmbeddingKernel')
        super().build(input_shape)

    def call(self, inputs):
        return self.embedding

    def compute_output_shape(self):
        return (1, self.num_latent_factors)

def get_matrix_factorization_model(
        num_items: int,
        num_latent_factors: int) -> tff.learning.models.ReconstructionModel:
    """Defines a Keras matrix factorization model."""
    # Layers with variables will be partitioned into global and local layers.
    # We'll pass this to `tff.learning.models.ReconstructionModel.from_keras_model_and_layers`.
    global_layers = []
    local_layers = []
    
    # Extract the item embedding.
    item_embedding_layer = tf.keras.layers.Embedding(
        num_items,
        num_latent_factors,
        name='ItemEmbedding')
    global_layers.append(item_embedding_layer)

    # Extract the user embedding.
    user_embedding_layer = UserEmbedding(
        num_latent_factors,
        name='UserEmbedding')
    local_layers.append(user_embedding_layer)

    item_input = tf.keras.layers.Input(shape=[1], name='Item')

    # The item_input never gets used by the user embedding layer,
    # but this allows the model to directly use the user embedding.
    flat_user_vec = user_embedding_layer(item_input)
    flat_item_vec = tf.keras.layers.Flatten(name='FlattenItems')(item_embedding_layer(item_input))

    # Compute the dot product between the user embedding, and the item one.
    pred = tf.keras.layers.Dot(1, normalize=False, name='Dot')([flat_user_vec, flat_item_vec])

    input_spec = collections.OrderedDict(
        x=tf.TensorSpec(shape=[None, 1], dtype=tf.int64),
        y=tf.TensorSpec(shape=[None, 1], dtype=tf.float32))
    
    model = tf.keras.Model(inputs=item_input, 
                                 outputs=pred)

    return tff.learning.models.ReconstructionModel.from_keras_model_and_layers(
        keras_model=model,
        global_layers=global_layers,
        local_layers=local_layers,
        input_spec=input_spec)
    
# Global variables used for keeping persistent training and evaluation processes between requests
eval_process = None
training_process = None
training_state = None
eval_state = None

def train_model(tf_train_datasets, tf_val_datasets):
    global eval_process
    global training_process
    global training_state
    global eval_state
    # Set the local Python execution context. This needs to be set to run TFF locally
    execution_contexts.set_sync_local_cpp_execution_context()
    
    if eval_process is None:
        model_fn = functools.partial(
            get_matrix_factorization_model,
            num_items=NUM_ITEMS,
            num_latent_factors=EMBEDDING_DIM)
            
        # Build the federated learning process
        loss_fn = lambda: tf.keras.losses.MeanSquaredError()
        # Update metrics_fn to include MAE and RMSE
        metrics_fn = lambda: [
            tf.keras.metrics.MeanAbsoluteError(name='mean_absolute_error'),
            tf.keras.metrics.RootMeanSquaredError(name='root_mean_squared_error')
        ]

        training_process = tff.learning.algorithms.build_fed_recon(
            model_fn=model_fn,
            loss_fn=loss_fn,
            metrics_fn=metrics_fn,
            server_optimizer_fn=tff.learning.optimizers.build_sgdm(LEARNING_RATE),
            client_optimizer_fn=tff.learning.optimizers.build_sgdm(LEARNING_RATE),
            reconstruction_optimizer_fn=tff.learning.optimizers.build_sgdm(LEARNING_RATE))
        
        eval_process = tff.learning.algorithms.build_fed_recon_eval(
            model_fn,
            loss_fn=loss_fn,
            metrics_fn=metrics_fn,
            reconstruction_optimizer_fn=tff.learning.optimizers.build_sgdm(LEARNING_RATE))

        # Initialize the training process state
        training_state = training_process.initialize()
        training_model_weights = training_process.get_model_weights(training_state)
        
        eval_state = eval_process.initialize()
        eval_process.set_model_weights(eval_state, training_model_weights)
    
    # Initialize lists to track metrics
    train_losses = []
    train_maes = []
    train_rmses = []
    val_losses = []
    val_maes = []
    val_rmses = []
    
    # Initialize early stopping variables
    best_val_loss = float('inf')
    best_training_state = None
    best_eval_state = None
    epochs_without_improvement = 0

    # Set your patience level (number of epochs to wait for improvement)
    patience = PATIENCE  # Ensure PATIENCE is defined globally or set it here

    num_available_clients = len(tf_train_datasets)
    num_clients_per_round = min(NUM_CLIENTS_PER_ROUND, num_available_clients)

    for i in range(NUM_EPOCHS):
        # Ensure clients have data
        federated_train_data = [dataset for dataset in tf_train_datasets if len(dataset) > 0]
        federated_train_data = np.random.choice(
            federated_train_data, size=num_clients_per_round, replace=False).tolist()
        
        training_state, metrics = training_process.next(training_state, federated_train_data)
        # Extract training metrics
        train_metrics = metrics['client_work']['train']
        train_losses.append(train_metrics.get('loss', 0))
        train_maes.append(train_metrics.get('mean_absolute_error', 0))
        train_rmses.append(train_metrics.get('root_mean_squared_error', 0))
        
        # Evaluate the model after each epoch
        eval_state = eval_process.set_model_weights(
            eval_state, training_process.get_model_weights(training_state))
        eval_state, eval_metrics = eval_process.next(eval_state, tf_val_datasets)
        # Extract validation metrics
        val_metrics = eval_metrics['client_work']['eval']['current_round_metrics']
        val_loss = val_metrics.get('loss', 0)
        val_mae = val_metrics.get('mean_absolute_error', 0)
        val_rmse = val_metrics.get('root_mean_squared_error', 0)
        val_losses.append(val_loss)
        val_maes.append(val_mae)
        val_rmses.append(val_rmse)
        
        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_training_state = training_state
            best_eval_state = eval_state
            epochs_without_improvement = 0
            print(f'Epoch {i+1}: Validation loss improved to {best_val_loss:.4f}')
        else:
            epochs_without_improvement += 1
            print(f'Epoch {i+1}: No improvement in validation loss for {epochs_without_improvement} epochs.')
            if epochs_without_improvement >= patience:
                print('Early stopping triggered.')
                # Restore the best model weights
                training_state = best_training_state
                eval_state = best_eval_state
                break
    
    # Prepare the metrics to return
    metrics = {
        'train_losses': train_losses,
        'train_maes': train_maes,
        'train_rmses': train_rmses,
        'val_losses': val_losses,
        'val_maes': val_maes,
        'val_rmses': val_rmses
    }
    
    pprint.pprint(metrics)
    
    return training_state, eval_state, metrics

# Store user permissions
user_permissions = {}

def tf_model_to_dataset(model):
    """Convert a TensorFlow model to a TFF dataset."""
    weights = model.get_weights()
    
    # Flatten and concatenate all weights into a single array
    flattened_weights = np.concatenate([w.flatten() for w in weights])
    
    # Create a dataset from the flattened weights
    dataset = tf.data.Dataset.from_tensor_slices(flattened_weights)
    
    # Batch the dataset
    return dataset.batch(100)  # Adjust batch size as needed

if __name__ == '__main__':
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(CERT_PATH, KEY_PATH)  # Your SSL certificates

    app.run(host='0.0.0.0', port=443, ssl_context=context)