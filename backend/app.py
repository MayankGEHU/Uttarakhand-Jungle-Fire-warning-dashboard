from flask import Flask, jsonify
from flask_cors import CORS
import requests
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)

# Load machine learning model
def load_model():
    model = joblib.load('model/model.pkl')  
    return model

API_KEY = "710b86a2f99df3be61bf8ce424b35544"  # Your OpenWeather API key
CITY = "Dehradun"
COUNTRY = "IN"

@app.route('/predict', methods=['GET'])
def predict_fire_risk():
    try:
        # Fetch current weather data
        url = f"https://api.openweathermap.org/data/2.5/weather?q={CITY},{COUNTRY}&appid={API_KEY}"
        response = requests.get(url)
        if response.status_code != 200:
            return jsonify({'error': 'Failed to fetch weather data'}), 500
        
        weather_data = response.json()

        # Extract features
        temperature = weather_data['main']['temp'] - 273.15  # Convert Kelvin to Celsius
        humidity = weather_data['main']['humidity']
        wind_speed = weather_data['wind']['speed']
        weather_desc = weather_data['weather'][0]['description']

        # Map weather description to numerical value
        weather_mapping = {'clear sky': 0, 'few clouds': 1, 'scattered clouds': 2, 'broken clouds': 3, 
                           'shower rain': 4, 'rain': 5, 'thunderstorm': 6, 'snow': 7, 'mist': 8}
        weather = weather_mapping.get(weather_desc, -1)

        # Prepare input for the model
        features = np.array([[temperature, humidity, weather, wind_speed]])

        # Load the model and predict
        model = load_model()
        prediction = model.predict(features)

        # Map prediction to risk labels
        risk_labels = {0: "Low Risk", 1: "High Risk"}
        risk = risk_labels.get(int(prediction[0]), "Unknown Risk")

        return jsonify({'prediction': risk, 'weather_data': {
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'weather': weather_desc
        }})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'Error making prediction'}), 500


if __name__ == '__main__':
    app.run(debug=True)
