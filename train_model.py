import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

# ---------- CONFIG ----------
RANDOM_STATE = 42
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

DATA_PATH = "data/telemetry.parquet"   # <--- CHANGE THIS

numeric_features = [
    'speed','roadspeed','speed_deviation','total_harshness',
    'battery_voltage_value','odometer','latitude','longitude',
    'harsh_count'
]

categorical_features = [
    'hour_of_day','day_of_week','municipality','suburb','town','road_present'
]

# ---------- FEATURE ENGINEERING ----------
def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Temporal features
    if 'unitdatetime' in df.columns:
        if not np.issubdtype(df['unitdatetime'].dtype, np.datetime64):
            df['unitdatetime'] = pd.to_datetime(df['unitdatetime'], errors='coerce')
        df['hour_of_day'] = df['unitdatetime'].dt.hour
        df['day_of_week'] = df['unitdatetime'].dt.day_name()
    else:
        df['hour_of_day'] = 0
        df['day_of_week'] = "Unknown"

    # Original columns expected from notebook
    for col in [
        'speed','roadspeed',
        'harsh_acceleration','hard_braking',
        'sharp_left_turn','sharp_right_turn',
        'road'
    ]:
        if col not in df.columns:
            if col == 'road':
                df[col] = ""
            else:
                df[col] = 0.0

    # Derived features
    df['speed_deviation'] = df['speed'] - df['roadspeed']
    df['total_harshness'] = (
        df['harsh_acceleration']
        + df['hard_braking']
        + df['sharp_left_turn']
        + df['sharp_right_turn']
    )
    df['road_present'] = (
        (~df['road'].isna()) & (df['road'] != '') & (df['road'] != '""')
    ).astype(int)
    df['harsh_count'] = df[[
        'harsh_acceleration','hard_braking',
        'sharp_left_turn','sharp_right_turn'
    ]].sum(axis=1)

    return df


def main():
    print(f"Loading data from: {DATA_PATH}")
    if DATA_PATH.lower().endswith(".parquet"):
        df = pd.read_parquet(DATA_PATH)
    else:
        df = pd.read_csv(DATA_PATH)

    print("Initial data shape:", df.shape)

    # Apply feature engineering
    df = feature_engineer(df)
    print("After feature engineering shape:", df.shape)

    if 'classification' not in df.columns:
        raise ValueError("Target column 'classification' not found in dataset.")

    X = df[numeric_features + categorical_features]
    y = df['classification']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Preprocessor
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    # Models
    models = {
        "RandomForest": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=RANDOM_STATE
            ))
        ]),
        "GradientBoosting": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', GradientBoostingClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE
            ))
        ]),
        "AdaBoost": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', AdaBoostClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE
            ))
        ])
    }

    cv_results = {}

    for name, pipeline in models.items():
        print(f"\n==============================")
        print(f"Training and evaluating: {name}")
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)
        n_classes = y_proba.shape[1]

        # ROC-AUC handling multi-class
        if n_classes == 2:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            y_test_bin = label_binarize(
                y_test,
                classes=pipeline.named_steps['classifier'].classes_
            )
            roc_auc = roc_auc_score(
                y_test_bin,
                y_proba,
                average='weighted',
                multi_class='ovr'
            )

        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC for {name}: {roc_auc:.3f}")
        cv_results[name] = {
            "pipeline": pipeline,
            "roc_auc": roc_auc
        }

    # Select best model
    best_model_name = max(cv_results, key=lambda k: cv_results[k]['roc_auc'])
    best_pipeline = cv_results[best_model_name]['pipeline']
    best_roc_auc = cv_results[best_model_name]['roc_auc']

    print("\n==============================")
    print(f"Best model: {best_model_name} with ROC-AUC: {best_roc_auc:.3f}")

    # Save best model
    model_path = os.path.join(MODEL_DIR, "best_pipeline.joblib")
    joblib.dump(best_pipeline, model_path)
    print(f"Best pipeline saved to {model_path}")


if __name__ == "__main__":
    main()
