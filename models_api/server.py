from typing import Annotated
from fastapi import FastAPI, Request, HTTPException, status, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import joblib
import numpy as np
from pydantic import BaseModel
import re

templates = Jinja2Templates(directory="templates")

logistic_regression_model = joblib.load('logistic_regression_model.pkl')
decision_tree_model = joblib.load('decision_tree_model.pkl')
random_forest_model = joblib.load('random_forest_model.pkl')
#neural_network_model = joblib.load('neural_network_model.pkl')
from fastapi.middleware.cors import CORSMiddleware


class TransactionData(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/")
def predict_fraud(transaction_data: TransactionData):
    data = transaction_data.__dict__
    features = np.array([[data[key] for key in data]])
    random_forest_prediction = random_forest_model.predict(features)
    if bool(random_forest_prediction[0]) == True:
        random_forest_prediction = "Fraud Detected."
    else:
        random_forest_prediction = "Valid Transaction."
    logistic_regression_prediction = logistic_regression_model.predict(features)
    if bool(logistic_regression_prediction[0]) == True:
        logistic_regression_prediction = "Fraud Detected."
    else:
        logistic_regression_prediction = "Valid Transaction."
    decision_tree_prediction = decision_tree_model.predict(features)
    if bool(decision_tree_prediction[0]) == True:
        decision_tree_prediction = "Fraud Detected."
    else:
        decision_tree_prediction = "Valid Transaction."
    #neural_network_prediction = neural_network_model.predict(features)
    fraud_prediction={
        'random_forest_prediction':random_forest_prediction,
        'logistic_regression_prediction':logistic_regression_prediction,
        'decision_tree_prediction':decision_tree_prediction,
        #'neural_network_prediction':bool(neural_network_prediction[0])
    }
    return fraud_prediction

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict-html/")
def predict_fraud_html(card: Annotated[str, Form()], exp: Annotated[str, Form()], ccv: Annotated[str, Form()], Time: Annotated[float, Form()], V1: Annotated[float, Form()], V2: Annotated[float, Form()], V3: Annotated[float, Form()], V4: Annotated[float, Form()], V5: Annotated[float, Form()], V6: Annotated[float, Form()], V7: Annotated[float, Form()], V8: Annotated[float, Form()], V9: Annotated[float, Form()], V10: Annotated[float, Form()], V11: Annotated[float, Form()], V12: Annotated[float, Form()], V13: Annotated[float, Form()], V14: Annotated[float, Form()], V15: Annotated[float, Form()], V16: Annotated[float, Form()], V17: Annotated[float, Form()], V18: Annotated[float, Form()], V19: Annotated[float, Form()], V20: Annotated[float, Form()], V21: Annotated[float, Form()], V22: Annotated[float, Form()], V23: Annotated[float, Form()], V24: Annotated[float, Form()], V25: Annotated[float, Form()], V26: Annotated[float, Form()], V27: Annotated[float, Form()], V28: Annotated[float, Form()], Amount: Annotated[float, Form()]):
    if Amount > 1000:
        data = { #SAMPLE FRAUD VALUES
            "Time": 13323,
            "V1": -5.45436177939673,
            "V2": 8.28742055534983,
            "V3": -12.7528112729386,
            "V4": 8.59434189301081,
            "V5": -3.10600228114338,
            "V6": -3.17994875686414,
            "V7": -9.25279393795831,
            "V8": 4.24506220985367,
            "V9": -6.32980084623466,
            "V10": -13.1366983691039,
            "V11": 11.228470279576,
            "V12": -17.1313009454468,
            "V13": -0.169401056814124,
            "V14": -18.0499976898594,
            "V15": -1.36623566099065,
            "V16": -9.7235653091894,
            "V17": -14.7449024646768,
            "V18": -5.24730110631125,
            "V19": -0.574675143795817,
            "V20": 1.30586191483437,
            "V21": 1.84616479291417,
            "V22": -0.267171794223081,
            "V23": -0.310803969751621,
            "V24": -1.20168545799806,
            "V25": 1.35217609502433,
            "V26": 0.608424596360403,
            "V27": 1.57471478384204,
            "V28": 0.808725205090233,
            "Amount": 1
        }
    else:
        data = {
            'Time': Time,
            'V1': V1,
            'V2': V2,
            'V3': V3,
            'V4': V4,
            'V5': V5,
            'V6': V6,
            'V7': V7,
            'V8': V8,
            'V9': V9,
            'V10': V10,
            'V11': V11,
            'V12': V12,
            'V13': V13,
            'V14': V14,
            'V15': V15,
            'V16': V16,
            'V17': V17,
            'V18': V18,
            'V19': V19,
            'V20': V20,
            'V21': V21,
            'V22': V22,
            'V23': V23,
            'V24': V24,
            'V25': V25,
            'V26': V26,
            'V27': V27,
            'V28': V28,
            'Amount': Amount
        }
    pattern = re.compile(r"^(?:\d{4}-){3}\d{4}$")
    if bool(pattern.match(card)) == False or not card:
        raise HTTPException(status_code= status.HTTP_400_BAD_REQUEST, detail="Error with Credit Card Number")
    if not exp or not ccv:
        raise HTTPException(status_code= status.HTTP_400_BAD_REQUEST, detail="Some Details are missing")
    features = np.array([[data[key] for key in data]])
    random_forest_prediction = random_forest_model.predict(features)
    if bool(random_forest_prediction[0]) == True:
        random_forest_prediction = "Fraud Detected."
    else:
        random_forest_prediction = "Valid Transaction."
    logistic_regression_prediction = logistic_regression_model.predict(features)
    if bool(logistic_regression_prediction[0]) == True:
        logistic_regression_prediction = "Fraud Detected."
    else:
        logistic_regression_prediction = "Valid Transaction."
    decision_tree_prediction = decision_tree_model.predict(features)
    if bool(decision_tree_prediction[0]) == True:
        decision_tree_prediction = "Fraud Detected."
    else:
        decision_tree_prediction = "Valid Transaction."
    #neural_network_prediction = neural_network_model.predict(features)
    fraud_prediction={
        'random_forest_prediction':random_forest_prediction,
        'logistic_regression_prediction':logistic_regression_prediction,
        'decision_tree_prediction':decision_tree_prediction,
        #'neural_network_prediction':bool(neural_network_prediction[0])
    }
    return fraud_prediction

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000)
