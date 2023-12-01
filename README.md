# House Price Prediction API (Flask)

## Description

This repository contains a Flask API for predicting house prices. It is designed to demonstrate how machine learning models can be integrated into web applications to provide real-time predictions. The project uses a Jupyter Notebook for the machine learning model, which is then incorporated into a Flask application to create an API endpoint.

## How to Use
To use this application, follow these steps:

### Prerequisites
- Python 3
- Pip (Python package manager)

### Installation
1. __Clone the Repository:__
```bash
git clone https://github.com/vmonney/House-price-prediction-api-flask.git
```

2. __Navigate to the Repository:__
```bash
cd House-price-predicition-api-flask
```

3. __Install Dependencies:__
```bash
pip install -r requirements.txt
```

### Running the Application

1. __Navigate to the app folder:__

```bash
cd app
```

2. __Start the Flask Server:__
```bash
uvicorn api:app --reload
```
This will start the Flas server, typically running on `http://127.0.0.1:8000`

3. __Accessing the API Endpoint:__
- The API endpoint for house price prediction is available at
`http://127.0.0.1:8000`.
- You can use tools like Postman or a simple curl command to send requests to this endpoint.

### Sending Requests
To make predictions, send a POST request to the `/` endpoint with the appropriate JSON payload containing the house features.

Example Request:

```bash
curl -X POST http://127.0.0.1:8000 -H "Content-Type: application/json" -d '{
"TransactionDate":"2020.12",
"HouseAge":40.0,
"DistanceToStation":467.6447748,
"NumberOfPubs":10.0,
"PostCode":"5222.0"
}'
```
