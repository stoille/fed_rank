import tensorflow as tf
import tensorflow_federated as tff
import collections

class UserEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_latent_factors: int, **kwargs):
        super().__init__(**kwargs)
        self.num_latent_factors = num_latent_factors

    def build(self, input_shape: tf.TensorShape) -> None:
        self.embedding = self.add_weight(
            shape=(1, self.num_latent_factors),
            initializer='uniform',
            dtype=tf.float32,
            name='UserEmbeddingKernel')
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.tile(self.embedding, [tf.shape(inputs)[0], 1])

def model_fn():
    # Define constants (set these to your actual numbers)
    num_users = 10000         # Total number of users
    num_articles = 10000      # Total number of articles
    embedding_dim = 32        # Embedding dimension (must be the same for both)

    # Define model inputs
    user_input = tf.keras.Input(shape=(1,), dtype=tf.int32, name='user_input')
    article_input = tf.keras.Input(shape=(1,), dtype=tf.int32, name='article_input')

    # Define embedding layers with the same output dimension
    user_embedding_layer = tf.keras.layers.Embedding(
        input_dim=num_users, output_dim=embedding_dim, name='user_embedding_layer'
    )
    article_embedding_layer = tf.keras.layers.Embedding(
        input_dim=num_articles, output_dim=embedding_dim, name='article_embedding_layer'
    )

    # Apply embedding layers to inputs
    user_embedding = user_embedding_layer(user_input)       # Shape: (None, 1, embedding_dim)
    article_embedding = article_embedding_layer(article_input)  # Shape: (None, 1, embedding_dim)

    # Flatten embeddings
    user_flatten_vec = tf.keras.layers.Flatten()(user_embedding)      # Shape: (None, embedding_dim)
    article_flatten_vec = tf.keras.layers.Flatten()(article_embedding)  # Shape: (None, embedding_dim)

    # Compute dot product between user and article embeddings
    pred = tf.keras.layers.Dot(axes=1, normalize=False, name='Dot')([user_flatten_vec, article_flatten_vec])
    input_spec = collections.OrderedDict(
            x=tf.TensorSpec(shape=(None, embedding_dim), dtype=tf.float32),
            y=tf.TensorSpec(shape=(None, embedding_dim), dtype=tf.float32))
    
    model = tf.keras.Model(inputs=[article_input, user_input], outputs=pred)
    
    return tff.learning.models.ReconstructionModel.from_keras_model_and_layers(
        keras_model=model,
        global_layers=[article_embedding_layer],
        local_layers=[user_embedding_layer],
        input_spec=input_spec,
    )