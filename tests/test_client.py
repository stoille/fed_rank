import unittest
import numpy as np
import tensorflow as tf
import json
from src.client import train_local_model, prepare_local_data, rank_articles

class TestClient(unittest.TestCase):

    def test_train_local_model(self):
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
        
        train_data = prepare_local_data(mock_candidate_articles)
        
        # Train the model
        updated_model = train_local_model(mock_model, train_data)

        # Check if the model was updated
        self.assertIsInstance(updated_model, tf.keras.Model)
        
        # Test ranking function separately
        ranked_results = rank_articles(updated_model, mock_candidate_articles)

        self.assertIsInstance(ranked_results, list)
        self.assertTrue(len(ranked_results) > 0)
        self.assertIsInstance(ranked_results[0], tuple)
        self.assertEqual(len(ranked_results[0]), 2)
        self.assertIsInstance(ranked_results[0][0], str)
        self.assertIsInstance(ranked_results[0][1], float)

if __name__ == '__main__':
    unittest.main()
