from fastapi import FastAPI, HTTPException, BackgroundTasks
from app.schema import IrisInput
import numpy as np
import joblib
import logging

# Load model
model = joblib.load("model/iris_model.pkl")

# Logging setup
logging.basicConfig(
    filename="api.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

app = FastAPI()

@app.post("/predict")
def predict(input_data: IrisInput, background_tasks: BackgroundTasks):
    try:
        data = np.array([[
            input_data.sepal_length,
            input_data.sepal_width,
            input_data.petal_length,
            input_data.petal_width
        ]])

        pred = model.predict(data)[0]
        proba = model.predict_proba(data)[0]

        species = ["setosa", "versicolor", "virginica"][pred]

        background_tasks.add_task(log_request, input_data, species)

        return {
            "prediction": species,
            "class_index": int(pred),
            "probabilities": {
                "setosa": float(proba[0]),
                "versicolor": float(proba[1]),
                "virginica": float(proba[2])
            }
        }

    except Exception:
        logging.exception("Error occurred")
        raise HTTPException(status_code=500, detail="Internal error")


def log_request(data: IrisInput, prediction: str):
    logging.info(f"Input: {data.dict()} | Prediction: {prediction}")