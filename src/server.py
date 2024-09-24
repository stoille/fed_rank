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

# global paths
CERT_PATH = os.path.join('data', 'cert.pem')
KEY_PATH = os.path.join('data', 'key.pem')
ARTICLES_PATH = os.path.join('data', 'articles.csv')
GLOBAL_MODEL_PATH = os.path.join('data', 'global_model.keras')
# Constants
NUM_EPOCHS = int(os.environ.get('NUM_EPOCHS', 10))
PATIENCE = int(os.environ.get('PATIENCE', 3))

app = Flask(__name__)

# Constants
NUM_USERS = int(os.getenv('NUM_USERS', '1000'))
NUM_ITEMS = int(os.getenv('NUM_ITEMS', '1000'))
EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', '32'))
MAX_EXAMPLES_PER_USER=300,
MAX_CLIENTS=2000
TOP_K = 10  # Define K for Precision@K and Recall@K
RANDOM_SEED = 42

# Global variables to store client updates
NUM_CLIENTS_PER_ROUND = 3  # Adjust this number based on your requirements
MAX_EXAMPLES_PER_USER = 100

@app.route('/feed', methods=['POST'])
def feed():
    print("initiating feed")
    global state
    global global_model
    global training_state
    global candidate_items
    
    # Load the ratings data
    articles_df, tf_train_datasets, tf_val_datasets = load_articles_and_ratings_data()  
    
    # Train the model
    # TODO: use eval metrics
    eval_state, eval_metrics = train_model(tf_train_datasets, tf_val_datasets)
    
    def get_keras_model():
        # This function should return a Keras model with the same architecture as your TFF model
        item_input = tf.keras.Input(shape=(1,), dtype=tf.int32, name='item_input')
        user_input = tf.keras.Input(shape=(1,), dtype=tf.int32, name='user_input')
        user_embedding = tf.keras.layers.Embedding(NUM_USERS, EMBEDDING_DIM)(user_input)
        user_embedding = tf.keras.layers.Flatten()(user_embedding)
        item_embedding = tf.keras.layers.Embedding(NUM_ITEMS, EMBEDDING_DIM)(item_input)
        item_embedding = tf.keras.layers.Flatten()(item_embedding)
        concatenated = tf.keras.layers.Concatenate()([user_embedding, item_embedding])
        dense = tf.keras.layers.Dense(64, activation='relu')(concatenated)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(dense)
        model = tf.keras.Model(inputs=[item_input, user_input], outputs=output)
        return model

    # Get keras model from TFF state
    keras_model = get_keras_model()
    tff.learning.models.ModelWeights.assign_weights_to(eval_state.global_model_weights, keras_model)

    # Save the model to a file
    keras_model.save(GLOBAL_MODEL_PATH)

    # Load the saved model file into a BytesIO buffer
    with open(GLOBAL_MODEL_PATH, 'rb') as f:
        model_buffer = io.BytesIO(f.read())

    # Reset the buffer's position to the beginning
    model_buffer.seek(0)

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
        'message': 'next POST /submit_updates to provide model updates and permission settings.'
    }

    # Attach the serialized model as a file in the response
    return send_file(
        model_buffer,
        as_attachment=True,
        download_name='global_model.keras',
        mimetype='application/octet-stream'
    ), 200, {'X-Response-Data': json.dumps(response_data)}

def load_articles_and_ratings_data(
    data_directory: str = "/tmp",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads pandas DataFrames for ratings, articles, users from data directory."""
    
    # Define the data types for each column
    dtype_ratings = {
        'UserID': int,
        'ItemID': int,
        'Rating': float,
        'Timestamp': str
    }
    # Load pandas DataFrames from data directory. Assuming data is formatted as
    # specified in UserID::ItemID::Rating::Timestamp
    ratings_df = pd.read_csv(
        os.path.join('data', "ratings.csv"),
        sep="::",
        dtype=dtype_ratings,
        names=["UserID", "ItemID", "Rating", "Timestamp"], engine="python")
    # Limit the number of clients to speed up dataset creation. 
    MAX_EXAMPLES_PER_USER = 1000
    tf_datasets = create_tf_datasets(
        ratings_df=ratings_df,
        batch_size=1000)
    
    # Load articles data
    # format: ItemId::Headline::Subheadline::Author::PublishDate::Category::ThumbnailUrl
    dtype_articles = {
        'ItemId': int,
        'Headline': str,
        'Subheadline': str,
        'Author': str,
        'PublishDate': str,
        'Category': str,
        'ThumbnailUrl': str
    }
    articles_df = pd.read_csv(
        os.path.join('data', "articles.csv"),
        sep="::",
        dtype=dtype_articles,
        names=["ItemId","Headline","Subheadline","Author","PublishDate","Category","ThumbnailUrl"], engine="python", 
        encoding = "ISO-8859-1")

    # Split the ratings into training/val/test by client.
    tf_train_datasets, tf_val_datasets, tf_test_datasets = split_tf_datasets(
        tf_datasets,
        train_fraction=0.8,
        val_fraction=0.1)

    return articles_df, tf_train_datasets, tf_val_datasets

def create_tf_datasets(ratings_df: pd.DataFrame,
                       batch_size: int = 1) -> List[tf.data.Dataset]:
    """Creates TF Datasets containing the articles and ratings for all users."""
    def rating_batch_map_fn(rating_batch):
        """Maps a rating batch to an OrderedDict with tensor values."""
        # Each example looks like: {x: article_id, y: rating}.
        # We won't need the UserID since each client will only look at their own
        # data.
        return collections.OrderedDict([
            ("x", tf.cast(rating_batch[:, 0:1], tf.int64)), #ItemID
            ("y", tf.cast(rating_batch[:, 1:2], tf.float32)) #Rating
        ])

    tf_datasets = []
    # Get the unique user IDs from the dataset
    unique_user_ids = ratings_df['UserID'].unique()
    
    for user_id in range(NUM_USERS):
        # Get subset of ratings_df belonging to a particular user.
        user_ratings_df = ratings_df[ratings_df.UserID == user_id]
        
        # Exclude any rows with missing values
        user_ratings_df = user_ratings_df[['ItemID', 'Rating']].dropna()

        tf_dataset = tf.data.Dataset.from_tensor_slices(user_ratings_df)

        # Define preprocessing operations.
        tf_dataset = tf_dataset.take(MAX_EXAMPLES_PER_USER).shuffle(
            buffer_size=MAX_EXAMPLES_PER_USER, seed=RANDOM_SEED).batch(batch_size).map(
            rating_batch_map_fn,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        tf_datasets.append(tf_dataset)

    return tf_datasets


def split_tf_datasets(
    tf_datasets: List[tf.data.Dataset],
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> Tuple[List[tf.data.Dataset], List[tf.data.Dataset], List[tf.data.Dataset]]:
    """Splits a list of user TF datasets into train/val/test by user.
    """
    np.random.seed(42)
    np.random.shuffle(tf_datasets)

    train_idx = int(len(tf_datasets) * train_fraction)
    val_idx = int(len(tf_datasets) * (train_fraction + val_fraction))

    # Note that the val and test data contains completely different users, not
    # just unseen ratings from train users.
    return (tf_datasets[:train_idx], tf_datasets[train_idx:val_idx],
            tf_datasets[val_idx:])

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
    item_input = tf.keras.layers.Input(shape=[1], name='Item')
    item_embedding_layer = tf.keras.layers.Embedding(
        num_items,
        num_latent_factors,
        name='ItemEmbedding')
    global_layers.append(item_embedding_layer)
    flat_item_vec = tf.keras.layers.Flatten(name='FlattenItems')(
        item_embedding_layer(item_input))

    # Extract the user embedding.
    user_embedding_layer = UserEmbedding(
        num_latent_factors,
        name='UserEmbedding')
    local_layers.append(user_embedding_layer)

    # The item_input never gets used by the user embedding layer,
    # but this allows the model to directly use the user embedding.
    flat_user_vec = user_embedding_layer(item_input)

    # Compute the dot product between the user embedding, and the item one.
    pred = tf.keras.layers.Dot(
        1, normalize=False, name='Dot')([flat_user_vec, flat_item_vec])

    input_spec = collections.OrderedDict(
        x=tf.TensorSpec(shape=[None, 1], dtype=tf.int64),
        y=tf.TensorSpec(shape=[None, 1], dtype=tf.float32))

    model = tf.keras.Model(inputs=item_input, outputs=pred)

    return tff.learning.models.ReconstructionModel.from_keras_model_and_layers(
        keras_model=model,
        global_layers=global_layers,
        local_layers=local_layers,
        input_spec=input_spec)


class RatingAccuracy(tf.keras.metrics.Mean):
    """Keras metric computing accuracy of reconstructed ratings."""

    def __init__(self,
                name: str='rating_accuracy',
                **kwargs):
        super().__init__(name=name, **kwargs)

    def update_state(self,
                    y_true: tf.Tensor,
                    y_pred: tf.Tensor,
                    sample_weight: Optional[tf.Tensor]=None):
        absolute_diffs = tf.abs(y_true - y_pred)
        # A [batch_size, 1] tf.bool tensor indicating correctness within the
        # threshold for each example in a batch. A 0.5 threshold corresponds
        # to correctness when predictions are rounded to the nearest whole
        # number.
        example_accuracies = tf.less_equal(absolute_diffs, 0.5)
        super().update_state(example_accuracies, sample_weight=sample_weight)

    loss_fn = lambda: tf.keras.losses.MeanSquaredError()
    metrics_fn = lambda: [RatingAccuracy()]

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
    
    if eval_process is None is None:
        # Check if running on M1/M2 Mac
        is_m1_mac = platform.processor() == 'arm'
        
        if is_m1_mac:
            optimizer = tf.keras.optimizers.legacy.SGD
        else:
            optimizer = tf.keras.optimizers.SGD
        
        model_fn = functools.partial(
            get_matrix_factorization_model,
            num_items=NUM_ITEMS,
            num_latent_factors=EMBEDDING_DIM)
            
        # Build the federated learning process
        loss_fn = lambda: tf.keras.losses.MeanSquaredError()
        metrics_fn = lambda: [RatingAccuracy()]

        training_process = tff.learning.algorithms.build_fed_recon(
            model_fn=model_fn,
            loss_fn=loss_fn,
            metrics_fn=metrics_fn,
            server_optimizer_fn=tff.learning.optimizers.build_sgdm(1.0),
            client_optimizer_fn=tff.learning.optimizers.build_sgdm(0.5),
            reconstruction_optimizer_fn=tff.learning.optimizers.build_sgdm(0.1))
        
        eval_process = tff.learning.algorithms.build_fed_recon_eval(
            model_fn,
            loss_fn=loss_fn,
            metrics_fn=metrics_fn,
            reconstruction_optimizer_fn=tff.learning.optimizers.build_sgdm(0.1))

        # Initialize the training process state
        training_state = training_process.initialize()
        training_model_weights = training_process.get_model_weights(training_state)
        
        eval_state = eval_process.initialize()
        eval_process.set_model_weights(eval_state, training_model_weights)
    
    train_losses = []
    train_accs = []
    
    # Train the model
    for i in range(NUM_EPOCHS):
        federated_train_data = np.random.choice(tf_train_datasets, size=50, replace=False).tolist()
        training_state, metrics = training_process.next(training_state, federated_train_data)
        print(f'Train round {i}:', metrics['client_work']['train'])
        train_losses.append(metrics['client_work']['train']['loss'])
        train_accs.append(metrics['client_work']['train']['rating_accuracy'])
        
    # Set the model weights in the evaluation process
    eval_state = eval_process.set_model_weights(
        eval_state, training_process.get_model_weights(training_state))
    # Evaluate the model
    eval_state, eval_metrics = eval_process.next(eval_state, tf_val_datasets)
    
    return eval_state, eval_metrics

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

client_datasets = []

@app.route('/submit_updates', methods=['POST'])
def submit_updates():
    global state
    global global_model
    global client_datasets
    
    execution_contexts.set_sync_local_cpp_execution_context()
    try:
        global state
        global global_model
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
            
            # Load the model with custom_objects if needed
            server_model = keras.models.load_model(
                temp_model_path,
                compile=False,
                custom_objects={'UserEmbedding': UserEmbedding}
            )
            
            # Compile the model with a desired optimizer and loss function
            server_model.compile(optimizer='sgd', loss='mean_squared_error')

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
            state, metrics = global_model.next(state, client_datasets)

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
    
    # Flatten and concatenate all weights into a single array
    flattened_weights = np.concatenate([w.flatten() for w in weights])
    
    # Create a dataset from the flattened weights
    dataset = tf.data.Dataset.from_tensor_slices(flattened_weights)
    
    # Batch the dataset
    return dataset.batch(1000)  # Adjust batch size as needed


if __name__ == '__main__':
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(CERT_PATH, KEY_PATH)  # Your SSL certificates

    app.run(host='0.0.0.0', port=443, ssl_context=context)
