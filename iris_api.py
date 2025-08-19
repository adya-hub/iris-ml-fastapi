from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import numpy as np
import logging
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

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

# HTML template setup
templates = Jinja2Templates(directory="templates")

# Root - serve HTML page
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# JSON API - predict
@app.post("/predict")
def predict(data: IrisInput):
    logger.info(f"Received input: {data.dict()}")
    input_data = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    prediction = model.predict(input_data)[0]
    labels = ['setosa', 'versicolor', 'virginica']
    logger.info(f"Prediction result: {labels[prediction]}")
    return {"prediction": labels[prediction]}

# Status endpoint
@app.get("/status")
def status():
    return {"status": "API is running", "uptime": "healthy"}
