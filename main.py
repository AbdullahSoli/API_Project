from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load the model and scaler
model_kmeans = joblib.load('kmens_model.joblib')
scaler_kmeans = joblib.load('kmens_scaler.joblib')

# Define the data model for the input
class InputFeatures(BaseModel):
    provider: str
    level: str
    type_: str
    duration_Weeks: str

# Function to preprocess the input data
def preprocess_features(input_features: InputFeatures):
    dict_f = {
        'provider': input_features.provider,
        'level': input_features.level,
        'type_': input_features.type_,
        'duration_Weeks': input_features.duration_Weeks
    }
    # Convert dictionary values to a list in the correct order
    features_list = [dict_f[key] for key in sorted(dict_f)]
    # Scale the input features
    scaled_features = scaler_kmeans.transform([features_list])
    return scaled_features

# Prediction endpoint
@app.post("/predict")
async def predict(input_features: InputFeatures):
    data = preprocess_features(input_features)
    y_pred = model_kmeans.predict(data)
    return {"prediction": y_pred.tolist()[0]}

@app.get("/")
def root():
    return "Prediction API is working."

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
