import os
import tensorflow as tf
import tensorflow_federated as tff

# Constants
NUM_USERS = int(os.getenv('NUM_USERS', '1000'))
NUM_ITEMS = int(os.getenv('NUM_ITEMS', '1000'))
EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', '32'))

def model_fn():
    """
    Creates and returns a TFF learning Model.
    
    Returns:
        A `tff.learning.Model` instance.
    """
    user_input = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name='user_input')
    item_input = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name='item_input')

    user_embedding = tf.keras.layers.Embedding(
        input_dim=NUM_USERS, output_dim=EMBEDDING_DIM, name='user_embedding')(user_input)
    item_embedding = tf.keras.layers.Embedding(
        input_dim=NUM_ITEMS, output_dim=EMBEDDING_DIM, name='item_embedding')(item_input)

    dot_product = tf.keras.layers.Dot(axes=-1)([user_embedding, item_embedding])
    output = tf.keras.layers.Flatten()(dot_product)

    # Optionally, add an activation function
    output = tf.keras.layers.Activation('sigmoid')(output)

    keras_model = tf.keras.Model(inputs=[user_input, item_input], outputs=output)

    # Define input specification
    input_spec = ({
        'user_input': tf.TensorSpec(shape=[None, 1], dtype=tf.int32),
        'item_input': tf.TensorSpec(shape=[None, 1], dtype=tf.int32)
    }, tf.TensorSpec(shape=[None, 1], dtype=tf.float32))

    return tff.learning.models.from_keras_model(
        keras_model=keras_model,
        input_spec=input_spec,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    ), keras_model

def tff_model_fn():
    keras_model, input_spec = model_fn()
    return tff.learning.models.from_keras_model(
        keras_model=keras_model,
        input_spec=input_spec,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

def get_keras_model():
    """
    Creates and returns the Keras model.
    """
    user_input = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name='user_input')
    item_input = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name='item_input')

    user_embedding = tf.keras.layers.Embedding(
        input_dim=NUM_USERS, output_dim=EMBEDDING_DIM, name='user_embedding')(user_input)
    item_embedding = tf.keras.layers.Embedding(
        input_dim=NUM_ITEMS, output_dim=EMBEDDING_DIM, name='item_embedding')(item_input)

    dot_product = tf.keras.layers.Dot(axes=-1)([user_embedding, item_embedding])
    output = tf.keras.layers.Flatten()(dot_product)

    # Add activation if necessary
    output = tf.keras.layers.Activation('sigmoid')(output)

    keras_model = tf.keras.Model(inputs=[user_input, item_input], outputs=output)
    keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return keras_model

# Test the functions
if __name__ == '__main__':
    # Test model_fn
    keras_model, input_spec = model_fn()
    keras_model.summary()

    # Test tff_model_fn
    tff_model = tff_model_fn()
    print("TFF Model created successfully.")