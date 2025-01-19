import requests
import pandas as pd
from datetime import datetime, timezone

# OpenWeather API Key and City details
api_key = "710b86a2f99df3be61bf8ce424b35544"
city = "Dehradun"
country = "IN"

# Fetch weather forecast data
url = f"https://api.openweathermap.org/data/2.5/forecast?q={city},{country}&appid={api_key}"
response = requests.get(url)

if response.status_code == 200:
    data = response.json()

    # Prepare dataset
    dates, avg_temperatures, avg_humidity, weather_descriptions, wind_speeds = [], [], [], [], []
    for entry in data['list']:
        timestamp = entry['dt']
        date = datetime.fromtimestamp(timestamp, timezone.utc).strftime('%Y-%m-%d')
        temperature = entry['main']['temp'] - 273.15
        humidity = entry['main']['humidity']
        description = entry['weather'][0]['description']
        wind_speed = entry['wind']['speed']

        if date not in dates:
            dates.append(date)
            avg_temperatures.append(temperature)
            avg_humidity.append(humidity)
            weather_descriptions.append(description)
            wind_speeds.append(wind_speed)
        else:
            index = dates.index(date)
            avg_temperatures[index] = (avg_temperatures[index] + temperature) / 2
            avg_humidity[index] = (avg_humidity[index] + humidity) / 2
            wind_speeds[index] = (wind_speeds[index] + wind_speed) / 2

    df = pd.DataFrame({
        'Date': dates,
        'Avg Temperature (°C)': avg_temperatures,
        'Avg Humidity (%)': avg_humidity,
        'Weather Description': weather_descriptions,
        'Wind Speed (m/s)': wind_speeds
    })

    # Save dataset
    df.to_csv("weather_data.csv", index=False)
    print("Weather data saved as weather_data.csv!")
else:
    print(f"Error fetching weather data: {response.status_code}")
