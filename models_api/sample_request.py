import requests

# Replace this URL with the actual URL of your FastAPI server
api_url = "http://localhost:8000/predict/"

# Sample transaction data (adjust the values according to your data)
sample_data = {
    "Time": 100.0,
    "V1": -1.358354,
    "V2": -1.340163,
    "V3": -0.073792,
    "V4": 2.329305,
    "V5": 0.744326,
    "V6": 0.188141,
    "V7": 0.651814,
    "V8": 0.069539,
    "V9": -0.175483,
    "V10": 0.739736,
    "V11": -0.247678,
    "V12": -1.514654,
    "V13": 0.243232,
    "V14": 0.364005,
    "V15": -0.082361,
    "V16": -0.470401,
    "V17": 0.207971,
    "V18": 0.025791,
    "V19": 0.403993,
    "V20": 0.251412,
    "V21": -0.018307,
    "V22": 0.277838,
    "V23": -0.110474,
    "V24": 0.066928,
    "V25": 0.128539,
    "V26": -0.189115,
    "V27": 0.133558,
    "V28": -0.021053,
    "Amount": 149.62
}

# Send the request to the API
response = requests.post(api_url, json=sample_data)

# Check the response
if response.status_code == 200:
    result = response.json()
    fraud_prediction = result["fraud_prediction"]
    print("Fraud Prediction:", fraud_prediction)
else:
    print("Error:", response.status_code)
