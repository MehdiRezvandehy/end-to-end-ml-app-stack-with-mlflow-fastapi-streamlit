import joblib
import pandas as pd
from datetime import datetime
from pydantic_objects import BuildingEnergyLoadPredictor, PredictionResponse

# Load model and preprocessor
MODEL_PATH = "./model/pickles/model.pkl"
SCALER_PATH = "./model/pickles/scaler.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model or preprocessor: {str(e)}")

def predict_proba(request: BuildingEnergyLoadPredictor) -> PredictionResponse:
    """
    Predict building energy load on using different building shapes.
    """
    # Prepare input data
    input_data = pd.DataFrame([request.model_dump()])

    # Rename columns to match training
    input_data.rename(columns={
        "rel_compact": "Relative Compactness",
        "surface_area": "Surface Area",
        "wall_area": "Wall Area",
        "roof_area": "Roof Area",
        "overall_height": "Overall Height",
        "orientation": "Orientation",
        "glazing_area": "Glazing Area",
        "glazing_dist": "Glazing Area Distribution"
    }, inplace=True)    

    # Preprocess input data
    scaled_features = scaler.transform(input_data)

    # Make prediction
    predicted_enery_load = model.predict_proba(scaled_features)[0][1]

    # Convert numpy.float32 to Python float and round to 2 decimal places
    predicted_enery_load = round(float(predicted_enery_load), 2)

    return PredictionResponse(
        predict_load_probability=predicted_enery_load,
    )
