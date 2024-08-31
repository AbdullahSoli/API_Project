from fastapi import FastAPI
app = FastAPI()
@app.get("/")
def root():
 return "Player Preduction"
@app.get("/items/{item_id}")
async def read_item(item_id):
 return {"item_id": item_id}


import joblib
model_kmeans = joblib.load('kmens_model.joblib')
scaler_kmeans = joblib.load('kmens_scaler.joblib')


from pydantic import BaseModel
class InputFeatures(BaseModel):
    Provider: str
    Level: str
    Type: str
    Duration_Weeks: str

def preprocessing(input_features: InputFeatures):
    dict_f = {
            'Provider': input_features.Provider ,
            'Level': input_features.Level, 
            'Type': input_features.Type, 
            'Duration / Weeks': input_features.Duration_Weeks ,
            
        }
    return dict_f


@app.get("/predict")
def predict(input_features: InputFeatures):
    return preprocessing(input_features)

def preprocessing(input_features: InputFeatures):
    dict_f = {
            'Provider': input_features.Provider ,
            'Level': input_features.Level, 
            'Type': input_features.Type, 
            'Duration / Weeks': input_features.Duration_Weeks ,
            
        }
    # Convert dictionary values to a list in the correct order
    features_list = [dict_f[key] for key in sorted(dict_f)]
    
    # Scale the input features
    scaled_features = scaler_kmeans.transform([list(dict_f.values
 ())])
    return scaled_features

@app.post("/predict")
async def predict(input_features: InputFeatures):
    data = preprocessing(input_features)
    y_pred = model_kmeans.predict(data)
    return {"pred": y_pred.tolist()[0]}



