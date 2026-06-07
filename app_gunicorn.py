# main.py
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel

app = FastAPI()

class PredictionInput(BaseModel):
    vendor_id: int
    passenger_count: int
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    store_and_fwd_flag: int
    pickup_hour: int
    pickup_dayofweek: int
    pickup_month: int
    pickup_weekofyear: int
    is_weekend: int
    is_night: int
    is_rush_hour: int

# Load the pre-trained RandomForest model
model_path = "models/model.joblib"
model = load(model_path)

@app.get("/")
def home():
    return "Working fine"

@app.post("/predict")
def predict(input_data: PredictionInput):
    features = [
        input_data.vendor_id,
        input_data.passenger_count,
        input_data.pickup_longitude,
        input_data.pickup_latitude,
        input_data.dropoff_longitude,
        input_data.dropoff_latitude,
        input_data.store_and_fwd_flag,
        input_data.pickup_hour,
        input_data.pickup_dayofweek,
        input_data.pickup_month,
        input_data.pickup_weekofyear,
        input_data.is_weekend,
        input_data.is_night,
        input_data.is_rush_hour,
    ]
    prediction = model.predict([features])[0].item()
    return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

# gunicorn -w 4 -k uvicorn.workers.UvicornWorker app_gunicorn:app
