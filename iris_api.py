from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import numpy as np
import logging
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Initialize FastAPI app
app = FastAPI()

# Load saved model
model = joblib.load("iris_model.pkl")

# Input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HTML template setup (templates folder must exist and contain index.html)
templates = Jinja2Templates(directory="templates")

# Root - serve HTML page
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# JSON API - predict
@app.post("/predict")
def predict(data: IrisInput):
    logger.info(f"Received input: {data.dict()}")

    # Convert input into numpy array
    input_data = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    
    # Model prediction
    prediction = model.predict(input_data)[0]
    labels = ['setosa', 'versicolor', 'virginica']
    result = labels[prediction]

    logger.info(f"Prediction result: {result}")
    return {"prediction": result}

# Status endpoint (for health check)
@app.get("/status")
def status():
    return {"status": "API is running", "uptime": "healthy"}
