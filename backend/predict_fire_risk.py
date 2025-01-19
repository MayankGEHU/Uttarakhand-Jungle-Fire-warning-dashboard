import pickle
import pandas as pd

rf_model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

new_data = pd.DataFrame({
    'Avg Temperature (°C)': [45],
    'Avg Humidity (%)': [15],
    'Weather Description': [0],  # Example: 0 = clear sky
    'Wind Speed (m/s)': [12]
})

scaled_data = scaler.transform(new_data)
prediction = rf_model.predict(scaled_data)

fire_risk = "High Risk" if prediction[0] == 1 else "Low Risk"
print("Fire Risk Prediction:", fire_risk)
