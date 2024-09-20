import sys
import tensorflow as tf
import tensorflow_federated as tff
import requests
import numpy as np
import json
import uuid
import collections
import io

def prepare_local_data(candidate_articles):
    # Map article IDs to indices
    article_id_to_index = {
        article['id']: idx for idx, article in enumerate(candidate_articles)
    }
    
    # Generate labels (e.g., random labels for demonstration)
    import random
    labels = [random.randint(0, 1) for _ in candidate_articles]
    
    # Prepare inputs
    article_indices = [
        article_id_to_index[article['id']] for article in candidate_articles
    ]
    article_lengths = [len(article['headline']) for article in candidate_articles]
    
    # Convert to tensors
    article_indices_tensor = tf.constant(article_indices, dtype=tf.int32)
    article_lengths_tensor = tf.constant(article_lengths, dtype=tf.float32)
    labels_tensor = tf.constant(labels, dtype=tf.float32)
    
    # Create a TensorFlow Dataset with the required inputs
    dataset = tf.data.Dataset.from_tensor_slices(
        ((article_indices_tensor, article_lengths_tensor), labels_tensor)
    )
    dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    
    return dataset

def train_local_model(model: tf.keras.Model, train_data):
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # Training loop
    for epoch in range(1):  # Adjust the number of epochs as needed
        for (batch_article_indices, batch_article_lengths), batch_labels in train_data:
            with tf.GradientTape() as tape:
                # Pass both inputs to the model
                predictions = model(
                    [batch_article_indices, batch_article_lengths], training=True
                )
                loss = loss_fn(batch_labels, predictions)
            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    # Prepare test data for inference
    article_id_to_index = {
        article['id']: idx for idx, article in enumerate(candidate_articles)
    }
    test_article_indices = [
        article_id_to_index[article['id']] for article in candidate_articles
    ]
    test_article_lengths = [len(article['headline']) for article in candidate_articles]
    
    # Convert to tensors
    test_article_indices_tensor = tf.constant(test_article_indices, dtype=tf.int32)
    test_article_lengths_tensor = tf.constant(test_article_lengths, dtype=tf.float32)
    
    # Get predictions
    predictions = model.predict(
        [test_article_indices_tensor, test_article_lengths_tensor]
    )

    # Create a list of (article_id, prediction) tuples and sort by prediction in descending order
    ranked_results = sorted(
        zip(
            [article['id'] for article in candidate_articles],
            predictions.flatten()
        ),
        key=lambda x: x[1],
        reverse=True
    )

    return model.get_weights(), ranked_results

if __name__ == '__main__':
    user_id = str(uuid.uuid4())

    # Initiate feed with the server
    try:
        feed_response = requests.post('https://localhost:443/feed', json={}, verify='cert.pem')

        # Extract the response data from headers
        response_data = json.loads(feed_response.headers.get('X-Response-Data', '{}'))
        
        # Extract candidate articles
        candidate_articles = response_data.get('candidate_articles', [])

        # Save the received model file
        with open('global_model.keras', 'wb') as f:
            f.write(feed_response.content)

        # Load the model without knowing its architecture
        model = tf.keras.models.load_model('global_model.keras')

    except Exception as err:
        print(f"An unexpected error occurred while fetching the feed: {err}")
        sys.exit(1)
        
    # Prepare local training data
    train_data = prepare_local_data(candidate_articles)

    # Ask user for permissions based on server's request in response_data (if necessary)
    permissions = json.dumps({"essential": True, "functional": True})
   
    # Train the local model and get updates and rankings
    model_updates, ranked_results = train_local_model(model, train_data)

    # Save the model to a file
    model.save('local_model.keras')

    # Load the saved model file into a BytesIO buffer
    with open('local_model.keras', 'rb') as f:
        model_buffer = io.BytesIO(f.read())

    # Reset the buffer's position to the beginning
    model_buffer.seek(0)

    # Prepare data to send to the server
    client_data = {
        'user_id': user_id,
        'permissions': permissions
    }

    # Send model updates and permissions to the server
    files = {
        'model': ('local_model.keras', model_buffer, 'application/octet-stream')
    }

    response = requests.post('https://localhost:443/submit_updates', 
                             data=client_data, 
                             files=files, 
                             verify='cert.pem')
    print(response.reason)

    # Display top 10 ranked results
    print("\nTop 10 Ranked Results:")
    for i, (index, score) in enumerate(ranked_results[:10], 1):
        print(f"{i}. Item {index}: Score {score:.4f}")
