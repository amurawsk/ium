# train_xgb_regressor_bin_300_1000.py

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin

from xgboost import XGBRegressor

# === Ścieżki
DATA_PATH = "../data/final_data.csv"
ATTRIBUTES_PATH = "utils/attributes.txt"
MODEL_OUTPUT_PATH = "final_models/xgb_regressor_bin_300_1000.pkl"

os.makedirs("final_models", exist_ok=True)

class PassthroughTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X

# === Wczytaj dane
df = pd.read_csv(DATA_PATH, sep=";")
df = df[(df["price"] >= df["price"].quantile(0.01)) & (df["price"] <= df["price"].quantile(0.99))].copy()

# Binning
bins = [0, 100, 300, 1000, np.inf]
labels = ["bin_0_100", "bin_100_300", "bin_300_1000", "bin_1000_up"]
df["price_bin"] = pd.cut(df["price"], bins=bins, labels=labels)

# Filtrowanie tylko dla bin_300_1000
df = df[df["price_bin"] == "bin_300_1000"].copy()
df["target"] = np.log1p(df["price"])

# Feature engineering (jak poprzednio)
from math import radians, sin, cos, sqrt, atan2
from sklearn.cluster import KMeans

def haversine(lat1, lon1, lat2=41.3870, lon2=2.1701):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

df["bathrooms_per_guest"] = df["bathrooms_num"] / df["accommodates"].replace(0, np.nan)
df["bedrooms_per_guest"] = df["bedrooms"] / df["accommodates"].replace(0, np.nan)
df["beds_per_guest"] = df["beds"] / df["accommodates"].replace(0, np.nan)
df["beds_per_bedroom"] = df["beds"] / df["bedrooms"].replace(0, np.nan)
df["guests_per_bedroom"] = df["accommodates"] / df["bedrooms"].replace(0, np.nan)
df["dist_to_center"] = df.apply(lambda row: haversine(row["latitude"], row["longitude"]), axis=1)

kmeans = KMeans(n_clusters=15, random_state=42)
df["location_cluster"] = kmeans.fit_predict(df[["latitude", "longitude"]])

amenity_cols = [col for col in df.columns if col.startswith("amenity_")]
df["num_amenities"] = df[amenity_cols].sum(axis=1)
df["has_tv_or_wifi"] = df[["amenity_TV", "amenity_Wifi"]].max(axis=1)
df["is_suited_for_longterm"] = df[["amenity_Washer", "amenity_Kitchen", "amenity_Dishes and silverware"]].sum(axis=1) >= 2
df["lat_scaled"] = (df["latitude"] - df["latitude"].mean()) / df["latitude"].std()
df["lon_scaled"] = (df["longitude"] - df["longitude"].mean()) / df["longitude"].std()
df["is_group_friendly"] = (df["accommodates"] >= 4).astype(int)

for col in ["bathrooms_per_guest", "bedrooms_per_guest", "beds_per_guest", "beds_per_bedroom", "guests_per_bedroom"]:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    q_low, q_high = df[col].quantile([0.01, 0.99])
    df[col] = df[col].clip(q_low, q_high).fillna(df[col].median())

# === Cechy
with open(ATTRIBUTES_PATH) as f:
    feature_cols = [line.strip() for line in f]

X = df[feature_cols]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

binary_cols = [col for col in X.select_dtypes(include="number").columns if X[col].nunique() <= 2]
numeric_cols = [col for col in X.select_dtypes(include="number").columns if col not in binary_cols]

numeric_pipeline = Pipeline([
    ("selector", PassthroughTransformer()),
    ("scaler", StandardScaler())
])
binary_pipeline = Pipeline([
    ("selector", PassthroughTransformer())
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("bin", binary_pipeline, binary_cols)
])

regressor = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

param_grid = {
    "reg__max_depth": [10, 15, 20],
    "reg__learning_rate": [0.005, 0.01, 0.02],
    "reg__n_estimators": [300, 500, 800],
    "reg__subsample": [0.7, 0.8, 1.0],
    "reg__colsample_bytree": [0.7, 0.9, 1.0]
}



pipeline = Pipeline([
    ("pre", preprocessor),
    ("reg", regressor)
])

grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="neg_root_mean_squared_error", verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)

# === Ewaluacja
best_model = grid.best_estimator_
y_pred_log = best_model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("=== XGBoost Metrics on bin_300_1000 ===")
print(f"MAE:  {mae:.2f} zł")
print(f"RMSE: {rmse:.2f} zł")
print(f"R²:   {r2:.4f}")
print("Best params:", grid.best_params_)

joblib.dump(best_model, MODEL_OUTPUT_PATH)
print(f"Zapisano XGBoost regresor do {MODEL_OUTPUT_PATH}")
