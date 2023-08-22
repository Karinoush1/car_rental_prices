import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from typing import Optional
import logging
from typing import List, Union
from fastapi import Query


description = """
### Getaround : Price Predictor 🚗
With this app you can easly set up an optimized price for your rental!
*  Fill in the data which is required
* The API will give you a prediction on the optimized price


### Endpoints:

1. **/predict/fields**: POST request. Fill in the required fields directly.
2. **/predict/json**: POST request. Provide your input in a JSON format.
3. **/**: A basic GET endpoint for a welcome message. 
"""

app = FastAPI(
    description = description, 
    title = "Car Rental Price API"
) 

# Define Pydantic model to parse incoming JSON data.
class PredictionFeatures(BaseModel):
    car_model_name: str 
    mileage :  int
    engine_power : int
    fuel : str
    car_type : str
    private_parking_available : bool
    has_gps: bool
    has_air_conditioning: bool
    automatic_car: bool
    has_getaround_connect: bool
    has_speed_regulator: bool
    winter_tires: bool


# Load model and preprocessor at startup
@app.on_event("startup")
def load_model():
    global model
    global preprocessor
    model = joblib.load('model_lr.pkl')
    preprocessor = joblib.load('preprocessor.pkl')

def predict_price(data):
    # Transform the data
    data = pd.DataFrame(data, index=[0])
    data = preprocessor.transform(data)

    # Predict
    prediction = model.predict(data)
    return prediction.tolist()

# endpoint which allows the user to fill in the parameters directly in the fields
@app.post("/predict/fields")
def predict_fields(
    car_model_name: str = Query(..., description="Name of the car's model (e.g., Citroën, Toyota, etc.)"),
    mileage: int = Query(..., description="Car's mileage"),
    engine_power: int = Query(..., description="Engine power of the car"),
    fuel: str = Query(..., description="Type of fuel used (e.g., diesel, petrol)"),
    car_type: str = Query(..., description="Type of the car (e.g., convertible, sedan)"),
    private_parking_available: bool = Query(..., description="Private parking is available"),
    has_gps: bool = Query(True, description="GPS is available"), 
    has_air_conditioning: bool = Query(..., description="Air conditioning is available"),
    automatic_car: bool = Query(..., description="Whether it is an automatic car"),
    has_getaround_connect: bool = Query(..., description="Getaround connect is available"),
    has_speed_regulator: bool = Query(..., description="Speed regulator is available"),
    winter_tires: bool = Query(..., description="Whether car has  winter tires"),
    
):
    
    data = {
        "car_model_name": car_model_name,
        "mileage": mileage,
        "engine_power": engine_power,
        "fuel": fuel,
        "car_type": car_type,
        "private_parking_available": private_parking_available,
        "has_gps": has_gps,
        "has_air_conditioning": has_air_conditioning,
        "automatic_car": automatic_car,
        "has_getaround_connect": has_getaround_connect,
        "has_speed_regulator": has_speed_regulator,
        "winter_tires": winter_tires
    }

    prediction = predict_price(data)
    return {"prediction": prediction}


#endpoint which allows user to provide input parameters in json format
@app.post("/predict/json")
class InputData(BaseModel):
    input: List[List[Union[str, int, float, bool]]]

@app.post("/predict")
def predict_from_input(data: InputData):
    features = data.input[0]
    
    # Map the features to the required format
    mapped_data = {
        "car_model_name": features[0],
        "mileage": features[1],
        "engine_power": features[2],
        "fuel": features[3],
        "car_type": features[4],
        "private_parking_available": bool(features[5]),
        "has_gps": bool(features[6]),
        "has_air_conditioning": bool(features[7]),
        "automatic_car": bool(features[8]),
        "has_getaround_connect": bool(features[9]),
        "has_speed_regulator": bool(features[10]),
        "winter_tires": bool(features[11])
    }
    
    prediction = predict_price(mapped_data)
    return {"prediction": prediction}

# Here you define endpoints 
@app.get('/')  # decorateur (=une fonction qui appelle une autre fonction) 
async def index():
    message = "Hello people! This is my first app, and you will be able to get car rental price predictions here! "
    return message

if __name__=="__main__":
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)



