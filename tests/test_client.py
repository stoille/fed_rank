import unittest
import tensorflow as tf
import json
from src.client import rank_items, prepare_local_data, train_local_model  # Add these imports

class TestClient(unittest.TestCase):

    def setUp(self):
        # Set up any necessary test data or configurations
        pass

    def test_train_local_model(self):
        # Create a simple mock model
        item_input = tf.keras.Input(shape=(1,), dtype=tf.int32, name='item_input')
        user_input = tf.keras.Input(shape=(1,), dtype=tf.int32, name='user_input')
        concatenated = tf.keras.layers.Concatenate()([item_input, user_input])
        output = tf.keras.layers.Dense(1)(concatenated)
        mock_model = tf.keras.Model(inputs={'item_input': item_input, 'user_input': user_input}, outputs=output)
        
        # Compile the mock model
        mock_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Create mock candidate articles in the same format as candidate_articles.json
        mock_candidate_articles = [
            {
                "id": f"item_{i}",
                "headline": f"Headline {i}",
                "category": "Technology",
                "author": f"Author {i}",
                "content": f"This is the content of article {i}.",
                "publication_date": "2023-05-01",
                "url": f"https://example.com/item_{i}"
            } for i in range(10)
        ]
        
        train_data, val_data = prepare_local_data(mock_candidate_articles)
        
        # Train the model
        updated_model = train_local_model(mock_model, train_data, val_data)[0]

        # Check if the model was updated
        self.assertIsInstance(updated_model, tf.keras.Model)
        
        # Test ranking function separately
        user_id = 1  # Add a sample user_id
        ranked_results = rank_items(updated_model, mock_candidate_articles, user_id)

        self.assertIsInstance(ranked_results, list)
        self.assertTrue(len(ranked_results) > 0)
        self.assertIsInstance(ranked_results[0], tuple)
        self.assertEqual(len(ranked_results[0]), 2)
        self.assertIsInstance(ranked_results[0][0], str)
        self.assertIsInstance(ranked_results[0][1], float)

    def test_rank_items(self):
        # Create a simple mock model
        item_input = tf.keras.Input(shape=(1,), dtype=tf.int32, name='item_input')
        user_input = tf.keras.Input(shape=(1,), dtype=tf.int32, name='user_input')
        concatenated = tf.keras.layers.Concatenate()([item_input, user_input])
        output = tf.keras.layers.Dense(1)(concatenated)
        mock_model = tf.keras.Model(inputs={'item_input': item_input, 'user_input': user_input}, outputs=output)

        # Create mock candidate articles
        mock_candidate_articles = [
            {
                "id": f"item_{i}",
                "headline": f"Headline {i}",
                "category": "Technology",
                "author": f"Author {i}",
                "content": f"This is the content of article {i}.",
                "publication_date": "2023-05-01",
                "url": f"https://example.com/item_{i}"
            } for i in range(10)
        ]

        # Mock user ID
        user_id = 1

        # Rank articles
        ranked_results = rank_items(mock_model, mock_candidate_articles, user_id)

        # Check if the results are correctly formatted
        self.assertIsInstance(ranked_results, list)
        self.assertGreater(len(ranked_results), 0)
        self.assertIsInstance(ranked_results[0], tuple)
        self.assertEqual(len(ranked_results[0]), 2)
        self.assertIsInstance(ranked_results[0][0], str)  # Article ID
        self.assertIsInstance(ranked_results[0][1], float)  # Score

        # Check if the results are sorted by score in descending order
        scores = [score for _, score in ranked_results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def tearDown(self):
        # Clean up any resources if needed
        pass

if __name__ == '__main__':
    unittest.main()
