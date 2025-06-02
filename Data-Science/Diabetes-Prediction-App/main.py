from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.model_loader import load_model_and_scaler
from app.predict import make_prediction

app = FastAPI()

# Define request model
class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Load model and scaler on startup
model, scaler = load_model_and_scaler()

@app.post("/predict")
def predict(data: DiabetesInput):
    try:
        # Convert to list format
        input_data = [
            data.Pregnancies, data.Glucose, data.BloodPressure,
            data.SkinThickness, data.Insulin, data.BMI,
            data.DiabetesPedigreeFunction, data.Age
        ]
        
        # Get prediction and probabilities
        prediction, probabilities = make_prediction(input_data, model, scaler)

        return {
            "prediction": prediction,
            "confidence": probabilities
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
