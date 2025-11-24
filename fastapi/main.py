from fastapi import FastAPI
from pydantic_objects import BuildingEnergyLoadPredictor, PredictionResponse
from fastapi.middleware.cors import CORSMiddleware
from run_prediction import predict_proba
from prometheus_fastapi_instrumentator import Instrumentator
import uvicorn

# Initialize FastAPI app with metadata
app = FastAPI(
    title="Building Energy Load Prediction API",
    description=(
        "An API for predicting building energy load based on various features using different building shapes. "
        "The data set is available on https://archive.ics.uci.edu/dataset/242/energy+efficiency. \n\n"
        "Authored by Mehdi Rezvandehy.\n "
    ),
    version="1.0.0",
    contact={
        "url": "https://mehdirezvandehy.github.io/",
        "email": "mrezvandehy@gmail.com",
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Health check endpoint
@app.get("/health", response_model=dict)
async def health_check():
    return {"status": "healthy", "model_loaded": True}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: BuildingEnergyLoadPredictor):
    return predict_proba(payload)

