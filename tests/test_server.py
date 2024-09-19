import unittest
import json
import os
import numpy as np
from server import app

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
        # Generate model updates with shapes matching the subset of server weights
        num_user_features = int(os.getenv('NUM_USER_FEATURES', '1000'))
        num_articles = int(os.getenv('NUM_ARTICLES', '1000'))
        embedding_dim = int(os.getenv('EMBEDDING_DIM', '32'))
        
        weight_shapes = [
            (num_articles, embedding_dim),  # Shape of article embedding layer
            (embedding_dim,)                # Shape of bias or other weights
        ]
        model_updates = [np.random.rand(*shape).tolist() for shape in weight_shapes]

        sample_data = {
            'user_id': 'test_user',
            'model_updates': json.dumps(model_updates),
            'permissions': {'essential': True}
        }
        response = self.app.post('/submit_updates', data=json.dumps(sample_data),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('metrics', data)

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
