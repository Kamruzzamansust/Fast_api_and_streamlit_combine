import streamlit as st
import requests

# Streamlit app
st.title("Car Price Prediction App")

# Input form for user to input values
years = st.slider("Years", min_value=1, max_value=10, step=1)
km = st.number_input("Kilometers", min_value=0.0, format="%.2f")
rating = st.slider("Rating", min_value=1.0, max_value=5.0, step=0.1)
condition = st.slider("Condition", min_value=1.0, max_value=5.0, step=0.1)
economy = st.slider("Economy", min_value=0.0, max_value=10.0, step=0.1)
top_speed = st.slider("Top Speed", min_value=0, max_value=300, step=1)
hp = st.slider("Horsepower", min_value=50, max_value=500, step=1)
torque = st.slider("Torque", min_value=50, max_value=500, step=1)

# Make prediction using FastAPI API
if st.button("Predict"):
    api_url = "http://127.0.0.1:8000/predict"
    payload = {
        "years": years,
        "km": km,
        "rating": rating,
        "condition": condition,
        "economy": economy,
        "top_speed": top_speed,
        "hp": hp,
        "torque": torque
    }

    try:
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            predicted_value = response.json()["current_price_prediction"]
            st.success(f"The predicted current price is: {predicted_value:.2f}")
        else:
            st.error(f"Error during prediction. Status Code: {response.status_code}")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
