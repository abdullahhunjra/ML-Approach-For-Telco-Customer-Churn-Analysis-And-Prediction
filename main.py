from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")
SCALER_PATH = os.getenv("SCALER_PATH")
ENCODERS_PATH = os.getenv("ENCODERS_PATH")

# Load model and transformers
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Load encoders for categorical features
encoder_internet = joblib.load(os.path.join(ENCODERS_PATH, "InternetService_encoder.pkl"))
encoder_security = joblib.load(os.path.join(ENCODERS_PATH, "OnlineSecurity_encoder.pkl"))
encoder_support  = joblib.load(os.path.join(ENCODERS_PATH, "TechSupport_encoder.pkl"))
encoder_contract = joblib.load(os.path.join(ENCODERS_PATH, "Contract_encoder.pkl"))

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Telco Churn Prediction API is running!"}

class CustomerData(BaseModel):
    tenure: float
    InternetService: str
    OnlineSecurity: str
    TechSupport: str
    Contract: str

@app.post("/predict")
def predict_churn(data: CustomerData):
    try:
        internet_encoded = encoder_internet.transform([data.InternetService])[0]
        security_encoded = encoder_security.transform([data.OnlineSecurity])[0]
        support_encoded  = encoder_support.transform([data.TechSupport])[0]
        contract_encoded = encoder_contract.transform([data.Contract])[0]
    except ValueError as e:
        return {
            "error": "Encoding error",
            "details": str(e),
            "tip": "Check for spelling or unseen category values."
        }

    try:
        tenure_scaled = scaler.transform([[data.tenure]])[0][0]
    except Exception as e:
        return {"error": f"Scaling error: {e}"}

    # Match the exact order of features during training
    final_input = np.array([[
        tenure_scaled,
        internet_encoded,
        security_encoded,
        support_encoded,
        contract_encoded
    ]])

    prediction = model.predict(final_input)[0]
    probability = model.predict_proba(final_input)[0][1]

    result = "Yes" if prediction == 1 else "No"

    return {
        "churn_prediction": result,
        "churn_probability": round(probability, 4)
    }
