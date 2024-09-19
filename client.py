import sys
import tensorflow as tf
import tensorflow_federated as tff
import requests
import numpy as np
import json
import uuid
import collections

def load_local_data() -> tf.data.Dataset:
    num_samples = 1000
    num_articles = 10000
    num_user_features = 10000

    # Generate integer IDs for articles and users
    article_ids = np.random.randint(0, num_articles, size=(num_samples, 1), dtype=np.int32)
    user_ids = np.random.randint(0, num_user_features, size=(num_samples, 1), dtype=np.int32)

    # Labels: 1 if user interacted positively, 0 otherwise
    labels = np.random.choice([0, 1], size=(num_samples, 1)).astype(np.float32)

    # Create features dictionary matching the model's expected inputs
    features = {
        'article_input': article_ids,
        'user_input': user_ids
    }

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    return dataset.shuffle(1000).batch(32)

def train_local_model(model: tf.keras.Model, permissions: dict) -> tuple[list[tf.Tensor], list[tuple[int, float]]]:
    # Load local data
    train_data = load_local_data()

    # Define optimizer and loss function
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # Perform local training
    for epoch in range(1):  # Adjust the number of epochs as needed
        for batch in train_data:
            with tf.GradientTape() as tape:
                predictions = model(batch[0], training=True)
                loss = loss_fn(batch[1], predictions)
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # Calculate the model updates (deltas), excluding UserEmbedding
    updated_weights = model.get_weights()
    model_updates = [w.tolist() for w in updated_weights]

    # Generate rankings (inference)
    test_data = load_local_data().take(100)
    predictions = model.predict(test_data)

    # Create a list of (index, prediction) tuples and sort by prediction in descending order
    ranked_results = sorted(enumerate(predictions.flatten()), key=lambda x: x[1], reverse=True)

    return model_updates, ranked_results

if __name__ == '__main__':
    user_id = str(uuid.uuid4())

    # Initiate feed with the server
    try:
        feed_response = requests.post('https://localhost:443/feed', json={}, verify='cert.pem')

        # Extract the response data from headers
        response_data = json.loads(feed_response.headers.get('X-Response-Data', '{}'))

        # Save the received model file
        with open('global_model.h5', 'wb') as f:
            f.write(feed_response.content)

        # Load the model without knowing its architecture
        model = tf.keras.models.load_model('global_model.h5')

    except Exception as err:
        print(f"An unexpected error occurred while fetching the feed: {err}")
        sys.exit(1)

    # Ask user for permissions based on server's request in response_data (if necessary)
    permissions = {'essential': True}
    # Handle permissions based on response_data['permissions_requested'], if provided

    # Train the local model and get updates and rankings
    model_updates, ranked_results = train_local_model(model, permissions)

    # Prepare data to send to the server
    client_data = {
        'user_id': user_id,
        'model_updates': json.dumps(model_updates),
        'permissions': permissions
    }

    # Send model updates and permissions to the server
    response = requests.post('https://localhost:443/submit_updates', json=client_data, verify='cert.pem')
    print(response.json())

    # Display top 10 ranked results
    print("\nTop 10 Ranked Results:")
    for i, (index, score) in enumerate(ranked_results[:10], 1):
        print(f"{i}. Item {index}: Score {score:.4f}")
