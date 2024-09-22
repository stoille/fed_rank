import unittest
import tensorflow as tf
import tensorflow_federated as tff
from src.shared_model import model_fn
from tensorflow_federated.python.core.backends.native import execution_contexts

class TestSharedModel(unittest.TestCase):

    def test_model_construction(self):
        tff_model_fn, tff_model = model_fn()
        self.assertIsInstance(tff_model, tf.keras.Model)
        # Check if it's a Functional model by ensuring it's a Model but not a Sequential
        self.assertIsInstance(tff_model, tf.keras.Model)
        self.assertNotIsInstance(tff_model, tf.keras.Sequential)
        
        # Set the local Python execution context
        execution_contexts.set_sync_local_cpp_execution_context()

    def test_model_forward_pass(self):
        _, model = model_fn()  # Get the Keras model

        batch_size = 4
        sample_inputs = {
            'item_input': tf.constant([[1], [2], [3], [4]], dtype=tf.int32),
            'user_input': tf.constant([[5], [6], [7], [8]], dtype=tf.int32)
        }

        predictions = model(sample_inputs, training=True)
        
        self.assertIsInstance(predictions, tf.Tensor)
        self.assertEqual(predictions.shape, (batch_size, 1))

if __name__ == '__main__':
    unittest.main()
