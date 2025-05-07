import pickle
import requests
import numpy as np
import os
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the model and LabelEncoders from pickle files
with open('model/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('model/le_soil.pkl', 'rb') as le_soil_file:
    le_soil = pickle.load(le_soil_file)

with open('model/le_season.pkl', 'rb') as le_season_file:
    le_season = pickle.load(le_season_file)

with open('model/le_irrigation.pkl', 'rb') as le_irrigation_file:
    le_irrigation = pickle.load(le_irrigation_file)

with open('model/le_crop.pkl', 'rb') as le_crop_file:
    le_crop = pickle.load(le_crop_file)

# Function to predict crop based on user input
def predict_crop(soil_type, season, farm_area, irrigation_type, water_usage):
    soil_type_encoded = le_soil.transform([soil_type])[0]
    season_encoded = le_season.transform([season])[0]
    irrigation_type_encoded = le_irrigation.transform([irrigation_type])[0]

    feature_array = np.array([[soil_type_encoded, season_encoded, farm_area, irrigation_type_encoded, water_usage]])
    predicted_crop_encoded = model.predict(feature_array)
    predicted_crop = le_crop.inverse_transform(predicted_crop_encoded)
    
    return predicted_crop[0]

# Function to get weather data
def get_weather(city):
    api_key = "1bd0be556664fb2dcaa474e56927746d"
    base_url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
    
    response = requests.get(base_url)
    data = response.json()

    if response.status_code != 200 or "main" not in data:
        return None

    return {
        "city": city,
        "Temperature": data["main"]["temp"],
        "Humidity": data["main"]["humidity"],
        "Weather": data["weather"][0]["description"],
        "Wind Speed": data["wind"]["speed"]
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input from the HTML form
    soil_type = request.form['soil_type']
    season = request.form['season']
    farm_area = float(request.form['farm_area'])
    irrigation_type = request.form['irrigation_type']
    water_usage = float(request.form['water_usage'])
    city = request.form['city']

    # Get predictions
    predicted_crop = predict_crop(soil_type, season, farm_area, irrigation_type, water_usage)
    weather_info = get_weather(city)

    return render_template('index.html', predicted_crop=predicted_crop, weather_info=weather_info)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
