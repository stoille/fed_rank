import sys
import tensorflow as tf
import tensorflow_federated as tff
import requests
import numpy as np
import json
import uuid
import collections
import io
import platform
import os

# Update the path for the certificate
CERT_PATH = os.path.join('data', 'cert.pem')
# Update paths for model files
GLOBAL_MODEL_PATH = os.path.join('data', 'global_model.keras')
LOCAL_MODEL_PATH = os.path.join('data', 'local_model.keras')

# Constants
NUM_EPOCHS = int(os.environ.get('NUM_EPOCHS', 10))
PATIENCE = int(os.environ.get('PATIENCE', 3))

def prepare_local_data(candidate_items):
    # This is a simplified version. You may need to adjust it based on your actual data structure
    item_ids = [hash(item['id']) % 10000 for item in candidate_items]
    labels = [1 if i % 2 == 0 else 0 for i in range(len(candidate_items))]  # Dummy labels
    
    # Split data into train and validation sets
    split = int(0.8 * len(item_ids))
    train_data = {
        'input': {'item_input': item_ids[:split], 'user_input': [1] * split},
        'output': labels[:split]
    }
    val_data = {
        'input': {'item_input': item_ids[split:], 'user_input': [1] * (len(item_ids) - split)},
        'output': labels[split:]
    }
    
    return train_data, val_data

def rank_items(model, candidate_items, user_id):
    item_input = tf.constant([[int(item['id'].split('_')[1])] for item in candidate_items])
    user_input = tf.constant([[user_id]] * len(candidate_items))

    predictions = model({'item_input': item_input, 'user_input': user_input})
    
    # Create a list of tuples (item_id, score)
    ranked_results = [(item['id'], float(score)) for item, score in zip(candidate_items, predictions.numpy().flatten())]
    
    # Sort the results by score in descending order
    ranked_results.sort(key=lambda x: x[1], reverse=True)
    
    return ranked_results

def train_local_model(model, train_data, val_data):
    # Compile the model if it's not already compiled
    if not model.compiled_loss:
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Prepare the input data
    train_item_input = np.array(train_data['input']['item_input'])
    train_user_input = np.array(train_data['input']['user_input'])
    train_output = np.array(train_data['output'])
    
    val_item_input = np.array(val_data['input']['item_input'])
    val_user_input = np.array(val_data['input']['user_input'])
    val_output = np.array(val_data['output'])
    
    # Train the model
    history = model.fit(
        {'item_input': train_item_input, 'user_input': train_user_input},
        train_output,
        validation_data=({'item_input': val_item_input, 'user_input': val_user_input}, val_output),
        epochs=NUM_EPOCHS,
        batch_size=32,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)]
    )
    
    return model, history

if __name__ == '__main__':
    user_id = str(uuid.uuid4())
    # Initiate feed with the server
    try:
        feed_response = requests.post('https://localhost:443/feed', json={}, verify=CERT_PATH)

        # Extract the response data from headers
        response_data = json.loads(feed_response.headers.get('X-Response-Data', '{}'))
        
        # Extract candidate items
        candidate_items = response_data.get('candidate_items', [])

        # Save the received model file
        with open(GLOBAL_MODEL_PATH, 'wb') as f:
            f.write(feed_response.content)

        # Load the model without knowing its architecture
        client_model = tf.keras.models.load_model(GLOBAL_MODEL_PATH, compile=False)
        client_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    except Exception as err:
        print(f"An unexpected error occurred while fetching the feed: {err}")
        sys.exit(1)
        
    # Prepare local training data
    train_data, val_data = prepare_local_data(candidate_items)

    # Ask user for permissions based on server's request in response_data (if necessary)
    permissions = json.dumps({"essential": True, "functional": True})
   
    # Train the model
    client_model, history = train_local_model(client_model, train_data, val_data)

    # Rank the items
    ranked_results = rank_items(client_model, candidate_items, user_id)

    # Save the model to a file
    client_model.save(LOCAL_MODEL_PATH)

    # Load the saved model file into a BytesIO buffer
    with open(LOCAL_MODEL_PATH, 'rb') as f:
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
                             verify=CERT_PATH)

    # Display top 10 ranked results
    print("\nTop 10 Ranked Results:")
    for i, (index, score) in enumerate(ranked_results[:10], 1):
        print(f"{i}. Item {index}: Score {score:.4f}")
