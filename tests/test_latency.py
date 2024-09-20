import unittest
import threading
import time
import requests
import json
import ssl
import uuid
import tensorflow as tf
from server import app
from client import prepare_local_data, train_local_model

class LatencyTests(unittest.TestCase):
    """
    Integration tests to measure end-to-end latency between client and server actions.
    """

    @classmethod
    def setUpClass(cls):
        """
        Start the server in a separate thread before running tests.
        """
        # Disable SSL warnings for testing purposes
        requests.packages.urllib3.disable_warnings()

        # Start the Flask server in a separate thread
        cls.server_thread = threading.Thread(target=cls.run_server)
        cls.server_thread.daemon = True
        cls.server_thread.start()

        # Wait a bit for the server to start
        time.sleep(2)

        # Set the server URL
        cls.server_url = 'https://localhost:443'

    @classmethod
    def run_server(cls):
        """
        Run the Flask server with SSL context.
        """
        # Set up SSL context for HTTPS
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain('cert.pem', 'key.pem')  # Your SSL certificates
        app.run(host='0.0.0.0', port=443, ssl_context=context, use_reloader=False)

    def test_end_to_end_latency(self):
        """
        Test the end-to-end latency of client actions interacting with the server.
        """
        # Step 1: Client requests the feed from the server
        start_time_feed = time.perf_counter()
        feed_response = requests.post(f'{self.server_url}/feed', json={}, verify=False)
        end_time_feed = time.perf_counter()
        feed_latency = end_time_feed - start_time_feed
        print(f"/feed endpoint latency: {feed_latency:.4f} seconds")

        self.assertEqual(feed_response.status_code, 200)
        self.assertIn('X-Response-Data', feed_response.headers)
        self.assertGreater(len(feed_response.content), 0)

        # Step 2: Client processes the feed and prepares training data
        response_data = json.loads(feed_response.headers.get('X-Response-Data', '{}'))
        candidate_articles = response_data.get('candidate_articles', [])

        # Save the received model file
        with open('global_model.keras', 'wb') as f:
            f.write(feed_response.content)
        
        # Load the model without knowing its architecture
        model = tf.keras.models.load_model('global_model.keras')

        # Prepare local training data
        prepare_start_time = time.perf_counter()
        train_data = prepare_local_data(candidate_articles)
        prepare_end_time = time.perf_counter()
        data_preparation_latency = prepare_end_time - prepare_start_time
        print(f"Data preparation latency: {data_preparation_latency:.4f} seconds")

        # Step 3: Client trains the local model
        train_start_time = time.perf_counter()
        model_updates = train_local_model(model, train_data)
        train_end_time = time.perf_counter()
        training_latency = train_end_time - train_start_time
        print(f"Local training latency: {training_latency:.4f} seconds")

        # Save the model to a file
        model.save('local_model.keras')

        # Load the saved model file into a buffer
        with open('local_model.keras', 'rb') as f:
            model_buffer = f.read()

        # Prepare data to send to the server
        user_id = str(uuid.uuid4())
        client_data = {
            'user_id': user_id,
            'permissions': json.dumps({"essential": True, "functional": True})
        }

        files = {
            'model': ('local_model.keras', model_buffer, 'application/octet-stream')
        }

        # Step 4: Client submits updates to the server
        submit_start_time = time.perf_counter()
        submit_response = requests.post(f'{self.server_url}/submit_updates',
                                        data=client_data,
                                        files=files,
                                        verify=False)
        submit_end_time = time.perf_counter()
        submit_latency = submit_end_time - submit_start_time
        print(f"/submit_updates endpoint latency: {submit_latency:.4f} seconds")

        self.assertEqual(submit_response.status_code, 200)

        # Total end-to-end latency
        total_latency = feed_latency + data_preparation_latency + training_latency + submit_latency
        print(f"Total end-to-end latency: {total_latency:.4f} seconds")

    @classmethod
    def tearDownClass(cls):
        """
        Perform any cleanup after tests.
        """
        # Note: Stopping the Flask server programmatically is not straightforward
        # The server runs in a daemon thread and will exit when the main program exits
        pass

if __name__ == '__main__':
    unittest.main()