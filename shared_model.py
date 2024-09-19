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

def model_fn(num_user_features=1000, num_articles=1000, embedding_dim=32):
    # Define model inputs
    article_input = tf.keras.Input(shape=(1,), dtype=tf.int32, name='article_input')
    user_input = tf.keras.Input(shape=(1,), dtype=tf.int32, name='user_input')

    # Define embedding layers
    article_embedding_layer = tf.keras.layers.Embedding(
        input_dim=num_articles, output_dim=embedding_dim, name='article_embedding_layer'
    )
    user_embedding_layer = tf.keras.layers.Embedding(
        input_dim=num_user_features, output_dim=embedding_dim, name='user_embedding_layer'
    )

    # Apply embeddings
    article_embedding = article_embedding_layer(article_input)
    user_embedding = user_embedding_layer(user_input)

    # Flatten embeddings
    article_flatten_vec = tf.keras.layers.Flatten()(article_embedding)
    user_flatten_vec = tf.keras.layers.Flatten()(user_embedding)

    # Compute dot product
    pred = tf.keras.layers.Dot(axes=1, normalize=False, name='Dot')([user_flatten_vec, article_flatten_vec])

    # Define the model
    model = tf.keras.Model(inputs={'article_input': article_input, 'user_input': user_input}, outputs=pred)

    # Update input_spec to match the client data
    input_spec = collections.OrderedDict(
        x=collections.OrderedDict(
            article_input=tf.TensorSpec(shape=[None, 1], dtype=tf.int32),
            user_input=tf.TensorSpec(shape=[None, 1], dtype=tf.int32)
        ),
        y=tf.TensorSpec(shape=[None, 1], dtype=tf.float32)
    )

    # Create the reconstruction model
    return tff.learning.models.ReconstructionModel.from_keras_model_and_layers(
        keras_model=model,
        global_layers=[article_embedding_layer],
        local_layers=[user_embedding_layer],
        input_spec=input_spec,
    ), model