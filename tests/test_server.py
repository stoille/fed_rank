import unittest
import json
import os
import numpy as np
import tensorflow as tf
from src.server import app, rank_articles
import io
import tempfile

class TestServer(unittest.TestCase):

    def setUp(self):
        """
        Set up the test client for the Flask app.
        """
        # Set environment variables for testing
        os.environ['NUM_USER_FEATURES'] = '1000'
        os.environ['NUM_ARTICLES'] = '1000'
        os.environ['EMBEDDING_DIM'] = '32'
        
        self.app = app.test_client()
        self.app.testing = True

        # Update model path
        self.temp_model_path = os.path.join('data', 'temp_model.keras')

    def test_feed_route(self):
        """
        Test the /feed route to ensure it returns the expected response.
        """
        response = self.app.post('/feed')
        self.assertEqual(response.status_code, 200)
        # Since feed returns a file, check headers
        self.assertIn('Content-Type', response.headers)
        self.assertIn('X-Response-Data', response.headers)

    def test_submit_updates_route(self):
        """
        Test the /submit_updates route with sample data.
        """
        # Create a simple mock Keras model
        num_user_features = int(os.getenv('NUM_USER_FEATURES', '1000'))
        num_articles = int(os.getenv('NUM_ARTICLES', '1000'))
        embedding_dim = int(os.getenv('EMBEDDING_DIM', '32'))

        article_input = tf.keras.Input(shape=(1,), dtype=tf.int32, name='article_input')
        user_input = tf.keras.Input(shape=(1,), dtype=tf.int32, name='user_input')
        
        article_embedding = tf.keras.layers.Embedding(num_articles, embedding_dim)(article_input)
        user_embedding = tf.keras.layers.Embedding(num_user_features, embedding_dim)(user_input)
        
        article_flatten = tf.keras.layers.Flatten()(article_embedding)
        user_flatten = tf.keras.layers.Flatten()(user_embedding)
        
        dot_product = tf.keras.layers.Dot(axes=1)([user_flatten, article_flatten])
        
        mock_model = tf.keras.Model(inputs=[article_input, user_input], outputs=dot_product)

        # Save the mock model to a temporary file
        mock_model.save(self.temp_model_path)

        # Prepare the data for the request
        with open(self.temp_model_path, 'rb') as model_file:
            sample_data = {
                'user_id': 'test_user',
                'permissions': json.dumps({'essential': True}),
                'model': (model_file, 'model.keras')
            }

            response = self.app.post('/submit_updates', 
                                     data=sample_data,
                                     content_type='multipart/form-data')
        
        # Clean up the temporary file
        os.remove(self.temp_model_path)
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('metrics', data)

    def test_rank_articles(self):
        # Create a simple mock model
        article_input = tf.keras.Input(shape=(1,), dtype=tf.int32, name='article_input')
        user_input = tf.keras.Input(shape=(1,), dtype=tf.int32, name='user_input')
        concatenated = tf.keras.layers.Concatenate()([article_input, user_input])
        output = tf.keras.layers.Dense(1)(concatenated)
        mock_model = tf.keras.Model(inputs={'article_input': article_input, 'user_input': user_input}, outputs=output)

        # Create mock candidate articles in the same format as candidate_articles.json
        mock_candidate_articles = [
            {
                "id": f"article_{i}",
                "headline": f"Headline {i}",
                "category": "Technology",
                "author": f"Author {i}",
                "content": f"This is the content of article {i}.",
                "publication_date": "2023-05-01",
                "url": f"https://example.com/article_{i}"
            } for i in range(10)
        ]

        # Rank articles
        ranked_results = rank_articles(mock_model, mock_candidate_articles)

        # Check if the results are correctly formatted
        self.assertEqual(len(ranked_results), 10)
        self.assertIsInstance(ranked_results[0], tuple)
        self.assertEqual(len(ranked_results[0]), 2)
        self.assertIsInstance(ranked_results[0][0], str)
        self.assertIsInstance(ranked_results[0][1], float)

    def tearDown(self):
        """
        Clean up after tests.
        """
        # Reset environment variables
        os.environ.pop('NUM_USER_FEATURES', None)
        os.environ.pop('NUM_ARTICLES', None)
        os.environ.pop('EMBEDDING_DIM', None)

if __name__ == '__main__':
    unittest.main()
