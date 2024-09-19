import unittest
import tensorflow as tf
import tensorflow_federated as tff
from shared_model import model_fn

class TestSharedModel(unittest.TestCase):

    def test_model_construction(self):
        """
        Test that the model_fn constructs a ReconstructionModel and a Keras model without errors.
        """
        reconstruction_model, keras_model = model_fn()
        self.assertIsInstance(reconstruction_model, tff.learning.models.ReconstructionModel)
        self.assertIsInstance(keras_model, tf.keras.Model)

    def test_model_forward_pass(self):
        """
        Test the model's forward pass with sample input data.
        """
        _, model = model_fn()  # Get the Keras model

        # Create sample inputs matching the input_spec
        batch_size = 4
        sample_inputs = {
            'article_input': tf.constant([[1], [2], [3], [4]], dtype=tf.int32),
            'user_input': tf.constant([[5], [6], [7], [8]], dtype=tf.int32)
        }

        # Perform a forward pass
        predictions = model(sample_inputs, training=True)
        
        self.assertIsInstance(predictions, tf.Tensor)
        self.assertEqual(predictions.shape, (batch_size, 1))

    def test_reconstruction_model_forward_pass(self):
        """
        Test the ReconstructionModel's forward pass with sample input data.
        """
        reconstruction_model, _ = model_fn()

        # Create sample inputs matching the input_spec
        batch_size = 4
        sample_inputs = {
            'article_input': tf.constant([[1], [2], [3], [4]], dtype=tf.int32),
            'user_input': tf.constant([[5], [6], [7], [8]], dtype=tf.int32)
        }
        sample_labels = tf.constant([[1.0], [0.0], [1.0], [0.0]], dtype=tf.float32)
        sample_batch = (sample_inputs, sample_labels)

        # Perform a forward pass
        output = reconstruction_model.forward_pass(sample_batch)
        
        #self.assertIsInstance(output.loss, tf.Tensor)
        self.assertIsInstance(output.predictions, tf.Tensor)
        self.assertEqual(output.predictions.shape, (batch_size, 1))
        self.assertEqual(output.labels.shape, (batch_size, 1))

if __name__ == '__main__':
    unittest.main()
