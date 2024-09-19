import sys
import tensorflow as tf
import tensorflow_federated as tff
import requests
import ssl
import numpy as np
from datetime import datetime, timedelta
import json
import uuid
import collections
from shared_model import model_fn

# Load feature configurations
with open('article_features.json', 'r') as f:
    article_feature_config = json.load(f)

with open('user_features.json', 'r') as f:
    user_feature_config = json.load(f)

def load_local_data() -> tf.data.Dataset:
    num_samples = 1000
    
    # Generate article features
    article_features = np.zeros((num_samples, len(article_feature_config)))
    for i, feature in enumerate(article_feature_config):
        if feature['type'] == 'categorical':
            article_features[:, i] = np.random.choice(feature['categories'], num_samples)
        elif feature['type'] == 'numerical':
            article_features[:, i] = np.random.uniform(feature['min'], feature['max'], num_samples)
    
    # Generate user features
    user_features = np.zeros((num_samples, len(user_feature_config)))
    for i, feature in enumerate(user_feature_config):
        if feature['type'] == 'categorical':
            user_features[:, i] = np.random.choice(feature['categories'], num_samples)
        elif feature['type'] == 'numerical':
            user_features[:, i] = np.random.uniform(feature['min'], feature['max'], num_samples)
    
    # Labels: 1 if user interacted positively, 0 otherwise
    labels = np.random.choice([0, 1], size=(num_samples, 1))
    
    dataset = tf.data.Dataset.from_tensor_slices(((article_features, user_features), labels))
    return dataset.shuffle(1000).batch(32)

def train_local_model(permissions: dict) -> tuple[list[tf.Tensor], list[tuple[int, float]]]:
    # Create and initialize the model
    model = model_fn()
    
    # Load local data
    train_data = load_local_data()
    
    # Define the client TFF computation
    @tff.tf_computation(model.trainable_variables_type, tf.data.Dataset.element_spec(train_data))
    def client_update(model_weights, dataset):
        model.set_weights(model_weights)
        for batch in dataset:
            with tf.GradientTape() as tape:
                loss = model.loss(y_true=batch[1], y_pred=model(batch[0]))
            grads = tape.gradient(loss, model.trainable_variables)
            for var, grad in zip(model.trainable_variables, grads):
                if 'UserEmbedding' not in var.name:  # Only update non-UserEmbedding weights
                    var.assign_sub(0.01 * grad)  # Simple SGD update
        return model.get_weights()
    
    # Perform local training
    initial_weights = model.initial_weights
    final_weights = client_update(initial_weights, train_data)
    
    # Calculate the model updates (deltas), excluding UserEmbedding
    model_updates = [
        update for var, update in zip(model.trainable_variables, tf.nest.map_structure(lambda a, b: a - b, final_weights, initial_weights))
        if 'UserEmbedding' not in var.name
    ]
    
    # Generate rankings
    test_data = load_local_data().take(100)  # Use a subset of data for testing
    model.set_weights(final_weights)
    predictions = model.predict(test_data)
    
    # Create a list of (index, prediction) tuples and sort by prediction in descending order
    ranked_results = sorted(enumerate(predictions.flatten()), key=lambda x: x[1], reverse=True)
    
    return model_updates, ranked_results

if __name__ == '__main__':
    user_id = str(uuid.uuid4())  # Generate a unique identifier for the user

    # Initiate feed with the server
    try:
        feed_response = requests.post('https://localhost:443/feed', json={}, verify='cert.pem')
        required_info = feed_response.json()
    except Exception as err:
        print(f"An unexpected error occurred while fetching the feed: {err}")
        # Handle any other exceptions
        sys.exit(1)
        
    # Ask user for permissions based on server's request
    # Ensure 'essential' is always True
    permissions = {'essential': True}
    for service, description in required_info.items():
        while True:
            user_input = input(f"Allow {service} ({description})? (Y/N): ").strip().lower()
            if user_input in ['y', 'n', 'Y', 'N']:
                permissions[service] = (user_input == 'y' or user_input == 'Y')
                break
            else:
                print("Invalid input. Please enter Y or N.")

    # Train the local model and get updates and rankings
    model_updates, ranked_results = train_local_model(permissions)

    # Serialize model updates
    serialized_updates = [update.numpy().tolist() for update in model_updates]

    # Prepare data to send to the server
    client_data = {
        'user_id': user_id,
        'model_updates': json.dumps(serialized_updates),
        'permissions': permissions
    }

    # Send model updates and permissions to the server
    response = requests.post('https://localhost:443/submit_updates', json=client_data, verify='cert.pem')
    print(response.json())

    # Display top 10 ranked results
    print("\nTop 10 Ranked Results:")
    for i, (index, score) in enumerate(ranked_results[:10], 1):
        print(f"{i}. Item {index}: Score {score:.4f}")
