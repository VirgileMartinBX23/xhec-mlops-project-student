# Pydantic models for the web service
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
                "predicted_age": 10.0
            }
        }
