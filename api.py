from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("my_saved_model")  # Update with your model directory name

# FastAPI app instance
app = FastAPI()

# Define the request body model using Pydantic
class Item(BaseModel):
    years: float
    km: float
    rating: float
    condition: float
    economy: float
    top_speed: float
    hp: float
    torque: float

# API endpoint to make predictions
@app.post("/predict")
def predict(item: Item):
    try:
        # Prepare the input features
        input_features = tf.convert_to_tensor([[item.years, item.km, item.rating,
                                                item.condition, item.economy, item.top_speed,
                                                item.hp, item.torque]], dtype=tf.float32)

        # Make predictions
        prediction = model.predict(input_features)

        # Extract the scalar prediction value
        predicted_value = prediction[0, 0].item()

        return {"current_price_prediction": predicted_value}

    except Exception as e:
        # Handle exceptions
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

# Add a welcome message at the root URL
@app.get("/")
def read_root():
    return {"message": "Welcome to the Car Price Prediction API!"}

# Run the FastAPI app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


