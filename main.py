from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load the model and scaler
model_kmeans = joblib.load('kmens_model.joblib')
scaler_kmeans = joblib.load('kmens_scaler.joblib')

# Define the data model for the input
class InputFeatures(BaseModel):
    Provider: str
    Level: str
    Type: str
    Duration_Weeks: str

# Function to preprocess the input data
def preprocess_features(input_features: InputFeatures):
    dict_f = {
        'Provider': input_features.Provider,
        'Level': input_features.Level,
        'Type': input_features.Type,
        'Duration_Weeks': input_features.Duration_Weeks
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
    return " Prediction"

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
