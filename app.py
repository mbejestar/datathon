import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os

# ---------- CONFIG ----------
MODEL_PATH = "models/best_pipeline.joblib"

numeric_features = [
    'speed', 'roadspeed', 'speed_deviation', 'total_harshness',
    'battery_voltage_value', 'odometer', 'latitude', 'longitude',
    'harsh_count'
]

categorical_features = [
    'hour_of_day', 'day_of_week', 'municipality', 'suburb', 'town', 'road_present'
]

# ---------- FEATURE ENGINEERING (same logic as notebook) ----------
def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure datetime
    if 'unitdatetime' in df.columns:
        if not np.issubdtype(df['unitdatetime'].dtype, np.datetime64):
            df['unitdatetime'] = pd.to_datetime(df['unitdatetime'], errors='coerce')
        df['hour_of_day'] = df['unitdatetime'].dt.hour
        df['day_of_week'] = df['unitdatetime'].dt.day_name()
    else:
        # If missing, create dummy time columns
        df['hour_of_day'] = 0
        df['day_of_week'] = "Unknown"

    # Derived features – requires these base columns (same names as in notebook)
    for col in ['speed', 'roadspeed',
                'harsh_acceleration', 'hard_braking',
                'sharp_left_turn', 'sharp_right_turn']:
        if col not in df.columns:
            df[col] = 0.0

    df['speed_deviation'] = df['speed'] - df['roadspeed']
    df['total_harshness'] = (
        df['harsh_acceleration']
        + df['hard_braking']
        + df['sharp_left_turn']
        + df['sharp_right_turn']
    )
    # road_present: 1 if a road name exists, 0 otherwise
    if 'road' in df.columns:
        df['road_present'] = (
            (~df['road'].isna()) & (df['road'] != '') & (df['road'] != '""')
        ).astype(int)
    else:
        df['road_present'] = 0

    df['harsh_count'] = df[[
        'harsh_acceleration', 'hard_braking',
        'sharp_left_turn', 'sharp_right_turn'
    ]].sum(axis=1)

    return df


# ---------- MODEL LOADER ----------
@st.cache_resource
def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            f"Make sure you've trained the model and saved it there."
        )
    model = joblib.load(model_path)
    return model


# ---------- STREAMLIT UI ----------
def main():
    st.title("🚗 Botho Driving – Harsh Event Detection")
    st.write(
        """
        Upload driver telemetry data to detect and analyse risky driving events
        such as harsh braking, harsh acceleration and sharp turns.
        """
    )

    st.sidebar.header("ℹ️ How to use")
    st.sidebar.markdown(
        """
        1. Train and save the model (see `train_model.py`).
        2. Make sure `models/best_pipeline.joblib` exists.
        3. Upload a **Parquet (.parquet)** or **CSV (.csv)** telemetry file.
        4. View predictions and summary statistics.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload telemetry file (.parquet or .csv)",
        type=["parquet", "csv"]
    )

    if uploaded_file is None:
        st.info("👆 Upload a telemetry file to get started.")
        return

    # Load data
    try:
        if uploaded_file.name.lower().endswith(".parquet"):
            df_raw = pd.read_parquet(uploaded_file)
        else:
            df_raw = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    st.subheader("Raw data preview")
    st.dataframe(df_raw.head())

    # Feature engineering
    df_fe = feature_engineer(df_raw)

    # Ensure required columns exist
    missing_cols = [
        c for c in (numeric_features + categorical_features) if c not in df_fe.columns
    ]
    if missing_cols:
        st.error(
            f"The following required features are missing after feature "
            f"engineering: {missing_cols}"
        )
        st.stop()

    X = df_fe[numeric_features + categorical_features]

    # Load model
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    # Predict
    with st.spinner("Running predictions..."):
        preds = model.predict(X)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
        else:
            proba = None

    df_results = df_fe.copy()
    df_results["prediction"] = preds

    # Add most probable class probability for quick reference
    if proba is not None:
        max_proba = proba.max(axis=1)
        df_results["prediction_confidence"] = max_proba

    st.subheader("Prediction results (first 20 rows)")
    st.dataframe(df_results.head(20))

    st.subheader("Prediction distribution")
    st.write(df_results["prediction"].value_counts())

    if proba is not None:
        st.subheader("Average prediction confidence")
        st.write(float(df_results["prediction_confidence"].mean()))

    # Allow download of results
    st.subheader("Download results")
    csv = df_results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download results as CSV",
        data=csv,
        file_name="batho_driving_predictions.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
