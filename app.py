import requests
import pandas as pd
from datetime import datetime
import joblib
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ====== CONFIG ======
GOOGLE_API_KEY = "YOUR_GOOGLE_MAPS_KEY"
WEATHER_API_KEY = "YOUR_OPENWEATHER_KEY"

# ====== FUNCTIONS ======

def get_traffic_data(origin, destination):
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        "origins": origin,
        "destinations": destination,
        "departure_time": "now",   # live traffic
        "key": GOOGLE_API_KEY
    }
    response = requests.get(url, params=params).json()

    try:
        duration = response["rows"][0]["elements"][0]["duration"]["value"]
        traffic_duration = response["rows"][0]["elements"][0]["duration_in_traffic"]["value"]
        distance = response["rows"][0]["elements"][0]["distance"]["value"]
        return {
            "duration": duration,
            "traffic_duration": traffic_duration,
            "distance": distance
        }
    except Exception as e:
        return None


def get_weather_data(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": WEATHER_API_KEY,
        "units": "metric"
    }
    response = requests.get(url, params=params).json()

    try:
        return {
            "temperature": response["main"]["temp"],
            "humidity": response["main"]["humidity"],
            "weather": response["weather"][0]["main"]
        }
    except:
        return None


def collect_data(origin, destination, lat, lon):
    traffic_info = get_traffic_data(origin, destination)
    weather_info = get_weather_data(lat, lon)

    if traffic_info and weather_info:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = {**traffic_info, **weather_info, "timestamp": timestamp}

        # Save into CSV
        df = pd.DataFrame([data])
        df.to_csv("traffic_weather_data.csv", mode="a", header=False, index=False)
        return data
    return None


def train_model():
    df = pd.read_csv("traffic_weather_data.csv", header=None)
    df.columns = ["duration", "traffic_duration", "distance", "temperature", "humidity", "weather", "timestamp"]

    # Encode weather (categorical -> numeric)
    df["weather_encoded"] = df["weather"].astype("category").cat.codes

    X = df[["distance", "temperature", "humidity", "weather_encoded"]]
    y = df["traffic_duration"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, "traffic_model.pkl")
    return model


def load_model():
    try:
        return joblib.load("traffic_model.pkl")
    except:
        return None


# ====== STREAMLIT WEB APP ======
st.title("🚦 AI-Powered Travel Time Predictor")

origin = st.text_input("Enter Origin", "VIT Vellore")
destination = st.text_input("Enter Destination", "Katpadi Railway Station")
lat = st.number_input("Latitude", value=12.9716)
lon = st.number_input("Longitude", value=79.1590)

if st.button("Collect Live Data"):
    data = collect_data(origin, destination, lat, lon)
    if data:
        st.success(f"Data Collected: {data}")
    else:
        st.error("Failed to fetch data")

if st.button("Train Model"):
    model = train_model()
    st.success("Model trained and saved!")

if st.button("Predict Travel Time"):
    model = load_model()
    if not model:
        st.error("Model not trained yet! Please train first.")
    else:
        traffic_info = get_traffic_data(origin, destination)
        weather_info = get_weather_data(lat, lon)

        if traffic_info and weather_info:
            weather_encoded = pd.Series([weather_info["weather"]]).astype("category").cat.codes[0]

            X_pred = pd.DataFrame([[
                traffic_info["distance"],
                weather_info["temperature"],
                weather_info["humidity"],
                weather_encoded
            ]], columns=["distance", "temperature", "humidity", "weather_encoded"])

            predicted_time = model.predict(X_pred)[0] / 60  # in minutes
            st.success(f"⏳ Predicted Travel Time: {predicted_time:.2f} minutes")
        else:
            st.error("Could not fetch live data for prediction.")
