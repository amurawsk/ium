import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# === Ścieżki
DATA_PATH = "../data/final_data.csv"
MODEL_PATH = "final_models/price_classifier.pkl"
OUTPUT_PATH = "bin_predictions.csv"

# === PassthroughTransformer (konieczny do odczytu modelu)
class PassthroughTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X

# === Wczytaj dane
df = pd.read_csv(DATA_PATH, sep=";")
df = df[(df["price"] >= df["price"].quantile(0.01)) & (df["price"] <= df["price"].quantile(0.99))].copy()

# === Feature engineering
df["bathrooms_per_guest"] = df["bathrooms_num"] / df["accommodates"].replace(0, np.nan)
df["bedrooms_per_guest"] = df["bedrooms"] / df["accommodates"].replace(0, np.nan)
df["beds_per_guest"] = df["beds"] / df["accommodates"].replace(0, np.nan)
df["beds_per_bedroom"] = df["beds"] / df["bedrooms"].replace(0, np.nan)
df["guests_per_bedroom"] = df["accommodates"] / df["bedrooms"].replace(0, np.nan)
df["dist_to_center"] = np.nan  # można zastąpić dokładną wartością jeśli masz

# Lokalizacja
try:
    kmeans = joblib.load("final_models/kmeans_location_cluster.pkl")
    df["location_cluster"] = kmeans.predict(df[["latitude", "longitude"]])
except FileNotFoundError:
    df["location_cluster"] = 0

# Dodatkowe cechy
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

# === Binning prawdziwej ceny
bins = [0, 100, 300, 1000, np.inf]
labels = ["bin_0_100", "bin_100_300", "bin_300_1000", "bin_1000_up"]
df["true_bin"] = pd.cut(df["price"], bins=bins, labels=labels)

# === Przygotowanie danych
X = df.drop(columns=["price", "id", "true_bin"])
if "price_bin" in X.columns:
    X = X.drop(columns=["price_bin"])

# === Wczytaj model
model = joblib.load(MODEL_PATH)

# === Predykcja
predicted_bins = model.predict(X)

# === Porównanie
output_df = df[["id", "price", "true_bin"]].copy()
output_df["predicted_bin"] = predicted_bins
output_df["correct"] = output_df["true_bin"] == output_df["predicted_bin"]

# === Eksport
output_df.to_csv(OUTPUT_PATH, index=False)
print(f"Zapisano wyniki do {OUTPUT_PATH}")
