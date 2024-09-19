Federated ranker privacy preserving personalization prototype. 

1. Generate a self-signed certificate & key:

openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365

2. Run the server & client via:

python server.py
python client.py
