import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pickle
import os
from datetime import datetime, timezone

df = pd.read_csv("weather_data.csv")

# Map weather descriptions to numeric values
weather_mapping = {'clear sky': 0, 'few clouds': 1, 'scattered clouds': 2, 'broken clouds': 3, 
                   'shower rain': 4, 'rain': 5, 'thunderstorm': 6, 'snow': 7, 'mist': 8}
df['Weather Description'] = df['Weather Description'].map(weather_mapping).fillna(-1)

# Define fire risk based on weather conditions 
df['Fire Risk'] = (df['Avg Temperature (°C)'] > 30) & (df['Avg Humidity (%)'] < 40) & (df['Wind Speed (m/s)'] > 5)
df['Fire Risk'] = df['Fire Risk'].astype(int)  

print(df['Fire Risk'].value_counts())

X = df[['Avg Temperature (°C)', 'Avg Humidity (%)', 'Weather Description', 'Wind Speed (m/s)']]
y = df['Fire Risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
print("Model Performance:\n", classification_report(y_test, y_pred))

# Save model, scaler, and metadata
os.makedirs("model", exist_ok=True)
pickle.dump(rf_model, open("model/model.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))

# Save model metadata
metadata = {
    "model_type": "RandomForestClassifier",
    "date_trained": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
    "features": list(X.columns)
}
pickle.dump(metadata, open("model/metadata.pkl", "wb"))

print("Model and scaler saved successfully!")

def predict_fire_risk(temperature, humidity, weather_description, wind_speed):
    weather_mapping = {'clear sky': 0, 'few clouds': 1, 'scattered clouds': 2, 'broken clouds': 3, 
                       'shower rain': 4, 'rain': 5, 'thunderstorm': 6, 'snow': 7, 'mist': 8}
    weather_value = weather_mapping.get(weather_description, -1)
    
    # Input data preprocessing
    input_data = pd.DataFrame([[temperature, humidity, weather_value, wind_speed]], columns=X.columns)
    input_data_scaled = scaler.transform(input_data)
    
    # Predict fire risk
    prediction = rf_model.predict(input_data_scaled)
    risk = "High risk of fire" if prediction[0] == 1 else "Low risk of fire"
    return risk

# Example prediction
temperature = 40  # Example input: 40°C
humidity = 15  # Example input: 15%
weather_description = "clear sky"  # Example input: clear sky
wind_speed = 25  # Example input: 25 m/s

print(predict_fire_risk(temperature, humidity, weather_description, wind_speed))
