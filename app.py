# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# -----------------------------
# Load the trained model
# -----------------------------
MODEL_PATH = "best_pipeline.joblib"
model = joblib.load(MODEL_PATH)

# -----------------------------
# Feature engineering function
# -----------------------------
def feature_engineer(df):
    df = df.copy()
    
    # Temporal features
    if 'unitdatetime' in df.columns:
        df['hour_of_day'] = df['unitdatetime'].dt.hour
        df['day_of_week'] = df['unitdatetime'].dt.day_name()
    
    # Derived features
    df['speed_deviation'] = df['speed'] - df['roadspeed']
    df['total_harshness'] = df['harsh_acceleration'] + df['hard_braking'] + df['sharp_left_turn'] + df['sharp_right_turn']
    df['road_present'] = (~df.get('road','').isna() & (df.get('road','') != '')).astype(int)
    df['harsh_count'] = df[['harsh_acceleration','hard_braking','sharp_left_turn','sharp_right_turn']].sum(axis=1)
    
    return df

# -----------------------------
# Prediction function
# -----------------------------
def predict_row(model, row_dict):
    # Convert input to DataFrame
    row_df = pd.DataFrame([row_dict])
    
    # Run feature engineering
    row_df = feature_engineer(row_df)
    
    # Keep only features expected by the model
    # Numeric + categorical names must match notebook
    numeric_features = ['speed','roadspeed','speed_deviation','total_harshness',
                        'battery_voltage_value','odometer','latitude','longitude','harsh_count']
    categorical_features = ['hour_of_day','day_of_week','municipality','suburb','town','road_present']
    
    X_row = row_df[numeric_features + categorical_features]
    
    # Predict class
    pred = model.predict(X_row)[0]
    
    # Predict probability for the predicted class
    try:
        proba_all = model.predict_proba(X_row)
        # Get the probability corresponding to predicted class
        class_index = list(model.named_steps['classifier'].classes_).index(pred)
        proba = proba_all[0][class_index]
    except Exception:
        proba = None
    
    return {'prediction': int(pred), 'probability': float(proba) if proba is not None else None}

# -----------------------------
# Streamlit app
# -----------------------------
st.title("Hazardous Road Segment Predictor")
st.write("Enter telematics data below to predict if a road segment is hazardous.")

# User inputs
speed = st.number_input("Vehicle Speed", 0, 200, 45)
roadspeed = st.number_input("Road Speed Limit", 0, 200, 60)
harsh_acceleration = st.number_input("Harsh Acceleration", 0, 1, 1)
hard_braking = st.number_input("Hard Braking", 0, 1, 1)
gforce_down = st.number_input("G-Force Down", 0, 5, 1)
sharp_left_turn = st.number_input("Sharp Left Turn", 0, 1, 0)
sharp_right_turn = st.number_input("Sharp Right Turn", 0, 1, 1)
latitude = st.number_input("Latitude", -90.0, 90.0, -26.2041)
longitude = st.number_input("Longitude", -180.0, 180.0, 28.0473)
odometer = st.number_input("Odometer Reading", 0, 1000000, 120000)
battery_voltage_value = st.number_input("Battery Voltage", 0.0, 20.0, 12.6)

municipality = st.text_input("Municipality", "OTHER")
suburb = st.text_input("Suburb", "OTHER")
town = st.text_input("Town", "OTHER")

current_hour = datetime.now().hour
current_day = datetime.now().strftime("%A")

# -----------------------------
# Prediction button
# -----------------------------
if st.button("Predict Hazard Level"):
    user_row = {
        'speed': speed,
        'roadspeed': roadspeed,
        'harsh_acceleration': harsh_acceleration,
        'hard_braking': hard_braking,
        'gforce_down': gforce_down,
        'sharp_left_turn': sharp_left_turn,
        'sharp_right_turn': sharp_right_turn,
        'latitude': latitude,
        'longitude': longitude,
        'odometer': odometer,
        'battery_voltage_value': battery_voltage_value,
        'municipality': municipality,
        'suburb': suburb,
        'town': town,
        'hour_of_day': current_hour,
        'day_of_week': current_day,
        'road': 'OTHER'  # default for single row
    }
    
    result = predict_row(model, user_row)
    
    st.subheader("Prediction Result")
    st.write(f"**Hazard Prediction (class):** {result['prediction']}")
    st.write(f"**Probability Score:** {result['probability']:.3f}")









