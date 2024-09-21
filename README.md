# Privacy Preserving Federated Ranker

Proof of concept privacy preserving collaborative filtering recommendation system. Building on [Tensorflow Federated](https://www.tensorflow.org/federated), the server and client independently train their own [Matrix Factorization](https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems)) models, using only a subset of shared latent factors. The server aggregates client model updates and averages them into the global model that future clients will use.

## Getting Started

1. Generate a self-signed certificate & key:

```
cd data
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365
```

2. Run the server & client via:

```
python src/server.py
python src/client.py
```

3. To run tests

```
python -m unittest discover tests
```
