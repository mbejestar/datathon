import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os

# -----------------------------
# Config & model loading
# -----------------------------
MODEL_PATH = "best_pipeline.joblib"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found. "
            f"Make sure it is in the same folder as app.py."
        )
    return joblib.load(MODEL_PATH)


# -----------------------------
# Feature engineering function
# -----------------------------
def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Temporal features (if unitdatetime present)
    if "unitdatetime" in df.columns:
        if not np.issubdtype(df["unitdatetime"].dtype, np.datetime64):
            df["unitdatetime"] = pd.to_datetime(df["unitdatetime"], errors="coerce")

        df["hour_of_day"] = df["unitdatetime"].dt.hour.fillna(0).astype(int)
        df["day_of_week"] = df["unitdatetime"].dt.day_name().fillna("Unknown")

    # If hour_of_day / day_of_week were already provided (single row),
    # we just leave them as-is.

    # Make sure required base columns exist
    for col in [
        "speed",
        "roadspeed",
        "harsh_acceleration",
        "hard_braking",
        "sharp_left_turn",
        "sharp_right_turn",
        "battery_voltage_value",
        "odometer",
        "latitude",
        "longitude",
    ]:
        if col not in df.columns:
            df[col] = 0.0

    # Derived features
    df["speed_deviation"] = df["speed"] - df["roadspeed"]
    df["total_harshness"] = (
        df["harsh_acceleration"]
        + df["hard_braking"]
        + df["sharp_left_turn"]
        + df["sharp_right_turn"]
    )

    # road_present: 1 if string exists, else 0
    if "road" in df.columns:
        road_series = df["road"].fillna("")
    else:
        road_series = pd.Series([""] * len(df))
        df["road"] = road_series  # keep it for consistency

    df["road_present"] = (road_series != "").astype(int)

    df["harsh_count"] = df[
        [
            "harsh_acceleration",
            "hard_braking",
            "sharp_left_turn",
            "sharp_right_turn",
        ]
    ].sum(axis=1)

    return df


# -----------------------------
# Prediction helper
# -----------------------------
def predict_row(model, row_dict):
    # Convert input to DataFrame
    row_df = pd.DataFrame([row_dict])

    # Run feature engineering
    row_df = feature_engineer(row_df)

    # Features used when training the model
    numeric_features = [
        "speed",
        "roadspeed",
        "speed_deviation",
        "total_harshness",
        "battery_voltage_value",
        "odometer",
        "latitude",
        "longitude",
        "harsh_count",
    ]
    categorical_features = [
        "hour_of_day",
        "day_of_week",
        "municipality",
        "suburb",
        "town",
        "road_present",
    ]

    # Ensure categorical columns exist even if left blank
    for col in categorical_features:
        if col not in row_df.columns:
            if col in ["hour_of_day"]:
                row_df[col] = 0
            elif col in ["day_of_week"]:
                row_df[col] = "Unknown"
            else:
                row_df[col] = "OTHER"

    X_row = row_df[numeric_features + categorical_features]

    # Predict class
    pred = model.predict(X_row)[0]

    # Predict probability for the predicted class (if available)
    proba = None
    try:
        proba_all = model.predict_proba(X_row)
        classes = model.named_steps["classifier"].classes_
        class_index = list(classes).index(pred)
        proba = float(proba_all[0][class_index])
    except Exception:
        proba = None

    return {"prediction": int(pred), "probability": proba}


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Hazardous Road Segment Predictor", page_icon="🚗")

st.title("🚗 Hazardous Road Segment Predictor")
st.write("Enter telematics data below to predict if a road segment is hazardous.")

with st.sidebar:
    st.header("ℹ️ Model info")
    st.write(
        """
        - Trained on telematics data  
        - Uses engineered features (speed deviation, harshness, etc.)  
        - Outputs a binary hazard prediction (0 = Non-hazard, 1 = Hazard)
        """
    )

# User inputs
speed = st.number_input("Vehicle Speed (km/h)", 0, 200, 45)
roadspeed = st.number_input("Road Speed Limit (km/h)", 0, 200, 60)
harsh_acceleration = st.number_input("Harsh Acceleration (0/1)", 0, 1, 1)
hard_braking = st.number_input("Hard Braking (0/1)", 0, 1, 1)
gforce_down = st.number_input("G-Force Down", 0.0, 5.0, 1.0)
sharp_left_turn = st.number_input("Sharp Left Turn (0/1)", 0, 1, 0)
sharp_right_turn = st.number_input("Sharp Right Turn (0/1)", 0, 1, 1)
latitude = st.number_input("Latitude", -90.0, 90.0, -26.2041)
longitude = st.number_input("Longitude", -180.0, 180.0, 28.0473)
odometer = st.number_input("Odometer Reading (km)", 0, 1_000_000, 120_000)
battery_voltage_value = st.number_input("Battery Voltage (V)", 0.0, 20.0, 12.6)

municipality = st.text_input("Municipality", "OTHER")
suburb = st.text_input("Suburb", "OTHER")
town = st.text_input("Town", "OTHER")

current_hour = datetime.now().hour
current_day = datetime.now().strftime("%A")

# Load model once
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -----------------------------
# Prediction button
# -----------------------------
if st.button("Predict Hazard Level"):
    user_row = {
        "speed": speed,
        "roadspeed": roadspeed,
        "harsh_acceleration": harsh_acceleration,
        "hard_braking": hard_braking,
        "gforce_down": gforce_down,  # not used by model but kept for completeness
        "sharp_left_turn": sharp_left_turn,
        "sharp_right_turn": sharp_right_turn,
        "latitude": latitude,
        "longitude": longitude,
        "odometer": odometer,
        "battery_voltage_value": battery_voltage_value,
        "municipality": municipality,
        "suburb": suburb,
        "town": town,
        "hour_of_day": current_hour,
        "day_of_week": current_day,
        "road": "OTHER",  # default road label for single-row input
    }

    result = predict_row(model, user_row)

    st.subheader("Prediction Result")
    st.write(f"**Hazard Prediction (class):** `{result['prediction']}`")

    if result["probability"] is not None:
        st.write(f"**Probability Score:** `{result['probability']:.3f}`")
    else:
        st.write("**Probability Score:** Not available for this model.")
