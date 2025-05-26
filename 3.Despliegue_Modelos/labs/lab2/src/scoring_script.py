import json
import os
import joblib
from azure.ml.inference.server import PredictionHandler

def init():
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.joblib")
    model = joblib.load(model_path)

def run(raw_data):
    data = json.loads(raw_data)
    predictions = model.predict(data["data"])
    return {"predictions": predictions.tolist()}