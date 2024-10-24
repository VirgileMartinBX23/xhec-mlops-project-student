from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel


# Input model
class AbaloneData(BaseModel):
    length: float
    diameter: float
    height: float
    whole_weight: float
    shucked_weight: float
    viscera_weight: float
    shell_weight: float
    sex: str  # "M" for Male, "F" for Female, "I" for Infant

    class Config:
        schema_extra = {
            "example": {
                "length": 0.5,
                "diameter": 0.4,
                "height": 0.1,
                "whole_weight": 0.5,
                "shucked_weight": 0.2,
                "viscera_weight": 0.1,
                "shell_weight": 0.15,
                "sex": "M"
            }
        }


# Output model
class AbalonePrediction(BaseModel):
    predicted_age: float

    class Config:
        schema_extra = {
            "example": {
                "predicted_ring": 10.0
            }
        }


app = FastAPI(title="Abalone Age Prediction API", description="Predict the age of abalone based on physical measurements")

MODEL_PATH = "src/web_service/local_objects/model.pkl"
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

@app.get("/")
def home() -> dict:
    return {"health_check": "App up and running!"}


@app.post("/predict", response_model="AbalonePrediction", status_code=201)
def predict(payload: "AbaloneData") -> AbalonePrediction:
    input_data = np.array([[payload.length, payload.diameter, payload.height, payload.whole_weight, 
                            payload.shucked_weight, payload.viscera_weight, payload.shell_weight, 
                            1 if payload.sex == 'M' else 0, 1 if payload.sex == 'F' else 0]])

    # Make a prediction
    predicted_age = model.predict(input_data)

    # Return the prediction in the expected response format
    return AbalonePrediction(predicted_age=predicted_age[0])