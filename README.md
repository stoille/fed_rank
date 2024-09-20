# Privacy Preserving Federated Ranker

This project aims to develop a proof of concept framework that empowers people to gain insight and set boundaries on how their personal data gets used. Recommender systems generaly overlook key aspects of data agency, ownership, and privacy. Despite the existence of regulations like GDPR and CCPA that seek to protect consumer data, there is an absence of standardized ranking solutions that inherently safeguard privacy. As a result, there is a pressing need to seek out methods that can address the issue of data provenance and user empowerment, enabling individuals to regain control over how their data is used. By ensuring personalized control and developing transparent systems for managing data, our goal is to develop privacy-preserving recommender systems that go beyond mere regulatory compliance and actively protect consumer privacy at a foundational level.

## Getting Started

1. Generate a self-signed certificate & key:

openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365

2. Run the server & client via:

python server.py
python client.py

3. To run tests
python -m unittest discover tests