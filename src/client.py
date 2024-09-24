import sys
import io
import os
import time
import json
import random
import requests
import platform
import numpy as np
import collections
import tensorflow as tf
import tensorflow_federated as tff
from collections import defaultdict

# Update the path for the certificate
CERT_PATH = os.path.join('data', 'cert.pem')
# Update paths for model files
GLOBAL_MODEL_PATH = os.path.join('data', 'global_model.keras')
LOCAL_MODEL_PATH = os.path.join('data', 'local_model.keras')
CANDIDATE_ITEMS_PATH = os.path.join('data', 'candidate_items.json')
# Constants
NUM_EPOCHS = int(os.environ.get('NUM_EPOCHS', 10))
PATIENCE = int(os.environ.get('PATIENCE', 3))

# Global variables to store interactions
explicit_interactions = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
implicit_interactions = defaultdict(lambda: defaultdict(float))

# Define explicit interaction types and their properties
EXPLICIT_INTERACTIONS = {
    'like': {'value': 1, 'cumulative': True},
    'dislike': {'value': -1, 'cumulative': True},
    'rating': {'value': None, 'cumulative': False},
    'share': {'value': 2, 'cumulative': True},
    'save': {'value': 1.5, 'cumulative': True},
}
# Global variables
user_data = None
user_feature_vector = None

def load_user_features_schema():
    with open('data/user_features_schema.json', 'r') as f:
        return json.load(f)

def generate_default_user(user_features_schema: list):
    default_user = {'id': random.randint(1, 1000)}
    
    for feature in user_features_schema:
        if feature['type'] == 'categorical':
            default_user[feature['name']] = random.choice(feature['categories'])
        elif feature['type'] == 'numerical':
            default_user[feature['name']] = random.uniform(feature['min'], feature['max'])
        elif feature['type'] == 'list':
            num_items = random.randint(feature['min_items'], feature['max_items'])
            default_user[feature['name']] = random.sample(feature['items'], num_items)
        # Add more types if needed
    
    return default_user

def generate_user_feature_vector(user_data, user_features_schema):
    feature_vector = []
    for feature in user_features_schema:
        if feature['type'] == 'categorical':
            feature_vector.extend(one_hot_encode(user_data[feature['name']], feature['categories']))
        elif feature['type'] == 'numerical':
            feature_vector.append(user_data[feature['name']])
        elif feature['type'] == 'list':
            feature_vector.extend([1 if item in user_data['name'] else 0 for item in feature['items']])
    return feature_vector

def one_hot_encode(value, categories):
    return [1 if value == category else 0 for category in categories]

def load_or_generate_user_data():
    global user_data, user_feature_vector
    user_data_path = os.path.join('data', 'user_data.json')
    user_features_schema = load_user_features_schema()
    
    if os.path.exists(user_data_path):
        with open(user_data_path, 'r') as f:
            user_data = json.load(f)
    else:
        user_data = generate_default_user(user_features_schema)
        with open(user_data_path, 'w') as f:
            json.dump(user_data, f)
    
    user_feature_vector = generate_user_feature_vector(user_data, user_features_schema)

def load_candidate_items():
    with open(CANDIDATE_ITEMS_PATH, 'r') as f:
        return json.load(f)

class UserItemInteraction(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(UserItemInteraction, self).__init__(**kwargs)

    def call(self, inputs):
        user_input, item_input = inputs
        # Expand user_input to match the batch size of item_input
        user_expanded = tf.expand_dims(user_input, axis=1)
        # Tile user_input to match the shape of item_input
        user_tiled = tf.tile(user_expanded, [1, tf.shape(item_input)[1], 1])
        # Concatenate user and item inputs
        return tf.concat([user_tiled, item_input], axis=-1)

def rank_items(model, user_id, candidate_items):
    # Prepare item_input as integer IDs
    item_ids = [item['ItemId'] for item in candidate_items]
    item_input = tf.constant(item_ids, dtype=tf.int32)
    item_input = tf.reshape(item_input, (len(item_ids), 1))

    # Prepare user_input by repeating the user_id to match the batch size
    user_ids = [int(user_id)] * len(item_ids)
    user_input = tf.constant(user_ids, dtype=tf.int32)
    user_input = tf.reshape(user_input, (len(user_ids), 1))

    # Get predictions
    predictions = model({'item_input': item_input, 'user_input': user_input})

    flat_predictions = predictions.numpy().flatten()

    # Create a list of tuples (item_id, score)
    ranked_results = [(item_id, float(score)) for item_id, score in zip(item_ids, flat_predictions)]

    # Sort the results by score in descending order
    ranked_results.sort(key=lambda x: x[1], reverse=True)
    
    return ranked_results
def prepare_local_data(candidate_items):
    # This is a simplified version. We may need to adjust it based on actual data structure
    item_ids = [hash(item['ItemId']) % 10000 for item in candidate_items]
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

def train_local_model(model, train_data, val_data):
    # Compile the model if it's not already compiled
    if not model.compiled_loss:
        model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    
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

def get_recommendations():
    try:
        feed_response = requests.post('https://localhost:443/feed', json={}, verify=CERT_PATH)
        feed_response.raise_for_status()  # Raise an exception for bad status codes
        
        response_data = json.loads(feed_response.headers.get('X-Response-Data', '{}'))
        candidate_items = response_data.get('candidate_items', [])

        with open(GLOBAL_MODEL_PATH, 'wb') as f:
            f.write(feed_response.content)

        client_model = tf.keras.models.load_model(GLOBAL_MODEL_PATH, compile=False)
        client_model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

        print(f"Received {len(candidate_items)} candidate items and updated global model.")
        return client_model, candidate_items
    except requests.RequestException as e:
        print(f"Error fetching recommendations: {e}")
        return None, None
    except Exception as err:
        print(f"An unexpected error occurred while fetching recommendations: {err}")
        return None, None

def interact_with_article(user_id, article_id, interaction_type, interaction_value=None):
    interaction_data = {
        'article_id': article_id,
        'interaction_type': interaction_type,
        'interaction_value': interaction_value,
        'timestamp': time.time()
    }
    
    if interaction_type in EXPLICIT_INTERACTIONS:
        interaction_props = EXPLICIT_INTERACTIONS[interaction_type]
        if interaction_value is None:
            interaction_value = interaction_props['value']
        
        if interaction_props['cumulative']:
            explicit_interactions[user_id][article_id][interaction_type] += interaction_value
        else:
            explicit_interactions[user_id][article_id][interaction_type] = interaction_value
    else:
        implicit_interactions[user_id][article_id] += (interaction_value or 1.0)
    
    print(f"Interaction with article {article_id} ({interaction_type}) recorded locally.")
    return interaction_data

def get_user_interactions(user_id):
    return explicit_interactions[user_id], implicit_interactions[user_id]

def update_global_model(client_model, user_id):
    try:
        client_model.save(LOCAL_MODEL_PATH)

        with open(LOCAL_MODEL_PATH, 'rb') as f:
            model_buffer = io.BytesIO(f.read())

        model_buffer.seek(0)

        client_data = {
            'user_id': user_id,
            'permissions': json.dumps({"essential": True, "functional": True})
        }

        files = {
            'model': ('local_model.keras', model_buffer, 'application/octet-stream')
        }

        response = requests.post('https://localhost:443/submit_updates', 
                                 data=client_data, 
                                 files=files, 
                                 verify=CERT_PATH)
        response.raise_for_status()
        
        print("Global model updated successfully")
    except requests.RequestException as e:
        print(f"Error updating global model: {e}")
    except Exception as err:
        print(f"An unexpected error occurred while updating the global model: {err}")

def interactive_client():
    global user_data
    user_id = user_data['id']
    client_model = None
    candidate_items = None
    
    print("Welcome to the interactive client. Type 'quit' at any time to exit.")

    while True:
        print("\nWhat would you like to do?")
        print("1. Get new recommendations")
        print("2. Interact with an article")
        print("3. Retrain client model")
        print("4. Update global model")
        print("5. View current interactions")
        print("6. Exit")

        choice = input("Enter your choice (1-6): ").lower()

        if choice == 'quit':
            print("Exiting...")
            break

        if choice == '1':
            client_model, candidate_items = get_recommendations()
            if client_model:
                ranked_results = rank_items(client_model, user_id, candidate_items)
                print("\nTop 10 Ranked Results:")
                for i, (item_id, score) in enumerate(ranked_results[:10], 1):
                    print(f"{i}. Item {item_id}: Score {score:.4f}")

        elif choice == '2':
            article_id = input("Enter the article ID you want to interact with: ")
            print("Available interaction types:")
            for i, interaction_type in enumerate(EXPLICIT_INTERACTIONS.keys(), 1):
                print(f"{i}. {interaction_type}")
            print(f"{len(EXPLICIT_INTERACTIONS) + 1}. Other (implicit)")
            
            interaction_choice = int(input("Choose an interaction type: "))
            if interaction_choice <= len(EXPLICIT_INTERACTIONS):
                interaction_type = list(EXPLICIT_INTERACTIONS.keys())[interaction_choice - 1]
                if interaction_type == 'rating':
                    interaction_value = float(input("Enter rating value: "))
                else:
                    interaction_value = EXPLICIT_INTERACTIONS[interaction_type]['value']
            else:
                interaction_type = input("Enter custom interaction type: ")
                interaction_value = float(input("Enter interaction value: "))
            
            interact_with_article(user_id, article_id, interaction_type, interaction_value)

        elif choice == '3':
            if not client_model or not candidate_items:
                print("Please get recommendations first.")
                continue
            train_data, val_data = prepare_local_data(candidate_items)
            client_model, history = train_local_model(client_model, train_data, val_data)
            print("Client model retrained successfully")

        elif choice == '4':
            if not client_model:
                print("Please get recommendations and retrain the client model first.")
                continue
            update_global_model(client_model, user_id)

        elif choice == '5':
            print("\nCurrent Interactions:")
            print("Explicit Interactions:")
            print(json.dumps(explicit_interactions[user_id], indent=2))
            print("\nImplicit Interactions:")
            print(json.dumps(implicit_interactions[user_id], indent=2))

        elif choice == '6':
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please try again.")

    print("Thank you for using the interactive client. Goodbye!")

if __name__ == '__main__':
    load_or_generate_user_data()
    interactive_client()