import joblib
import numpy as np
import pandas as pd
import os
from sklearn.neighbors import BallTree

# ===============================
# 1. LOAD ARTIFACTS
# ===============================

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(MODEL_DIR, "XGBoot_model.pkl"))
encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
config = joblib.load(os.path.join(MODEL_DIR, "preprocess_config.pkl"))

# Load POI
POI_PATH = os.path.join(BASE_DIR, "data", "smart_poi_data.csv")
poi_df = pd.read_csv(POI_PATH)
poi_df["lat"] = pd.to_numeric(poi_df["lat"], errors="coerce")
poi_df["lon"] = pd.to_numeric(poi_df["lon"], errors="coerce")

# ===============================
# 2. SPATIAL FEATURE ENGINEERING
# ===============================

def add_nearest_distance(df, type_name, new_col_name):
    poi_subset = poi_df[poi_df["type"] == type_name]

    if poi_subset.empty:
        df[new_col_name] = -1
        return df

    house_rad = np.radians(df[["lat", "lon"]].values)
    poi_rad = np.radians(poi_subset[["lat", "lon"]].values)

    tree = BallTree(poi_rad, metric="haversine")
    dist, _ = tree.query(house_rad, k=1)

    df[new_col_name] = dist * 6371  # km
    return df


def add_density_count(df, type_name, radius_km, new_col_name):
    poi_subset = poi_df[poi_df["type"] == type_name]

    if poi_subset.empty:
        df[new_col_name] = 0
        return df

    house_rad = np.radians(df[["lat", "lon"]].values)
    poi_rad = np.radians(poi_subset[["lat", "lon"]].values)

    tree = BallTree(poi_rad, metric="haversine")
    radius_rad = radius_km / 6371

    counts = tree.query_radius(house_rad, r=radius_rad, count_only=True)
    df[new_col_name] = counts
    return df


def build_spatial_features(df):
    df = add_nearest_distance(df, "Chợ", "dist_nearest_market")
    df = add_nearest_distance(df, "Bệnh viện", "dist_nearest_hospital")
    df = add_nearest_distance(df, "Trường đại học", "dist_nearest_university")

    df = add_density_count(df, "Trường đại học", 3.0, "count_uni_in_3km")
    df = add_density_count(df, "Bệnh viện", 5.0, "count_hospital_in_5km")
    
    return df


# ===============================
# 3. PREPROCESS FOR INFERENCE
# ===============================

def preprocess_for_prediction(df_input: pd.DataFrame) -> pd.DataFrame:

    X = df_input.copy()

    # ---- Log transform ----
    for col in config["log_features"]:
        if col in X.columns:
            X[col] = np.log1p(X[col])

    # ---- Encode categorical ----
    for col, le in encoders.items():
        if col in X.columns:
            X[col] = X[col].astype(str)

            unseen = set(X[col]) - set(le.classes_)
            if unseen:
                X[col] = X[col].apply(
                    lambda x: "UNKNOWN" if x in unseen else x
                )

                if "UNKNOWN" not in le.classes_:
                    le.classes_ = np.append(le.classes_, "UNKNOWN")

            X[col] = le.transform(X[col])

    # ---- Ensure feature order ----
    X = X.reindex(columns=config["feature_order"], fill_value=0)

    return X


# ===============================
# 4. PREDICT FUNCTION (FINAL)
# ===============================

def predict_price(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Raw input → spatial features → preprocess → predict
    """

    df = df_input.copy()

    # 1. Build spatial features from lat/lon
    df = build_spatial_features(df)

    # 2. Preprocess (log + encode + order)
    X_processed = preprocess_for_prediction(df)

    # 3. Predict (log scale)
    y_pred_log = model.predict(X_processed)

    # 4. Inverse log
    y_pred = np.expm1(y_pred_log)

    result = df_input.copy()
    result["predicted_price"] = y_pred
    
    return result
