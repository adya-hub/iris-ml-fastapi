from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load saved model
model = joblib.load("iris_model.pkl")

# Input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def root():
    return {"message": "API working"}

@app.post("/predict")
def predict(data: IrisInput):
    input_data = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    prediction = model.predict(input_data)[0]
    labels = ['setosa', 'versicolor', 'virginica']
    return {"prediction": labels[prediction]}

