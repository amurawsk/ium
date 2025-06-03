# evaluate_full_pipeline.py
import pandas as pd
import numpy as np
import joblib
from math import radians, sin, cos, sqrt, atan2
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin

# === PassthroughTransformer (konieczny do odczytu modelu)
class PassthroughTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X

# === Ścieżki
DATA_PATH = "../data/final_data.csv"
CLASSIFIER_PATH = "final_models/price_classifier.pkl"
REGRESSOR_PATHS = {
    "bin_0_100": "final_models/xgb_regressor_bin_0_100.pkl",
    "bin_100_300": "final_models/xgb_regressor_bin_100_300.pkl",
    "bin_300_1000": "final_models/xgb_regressor_bin_300_1000.pkl",
    "bin_1000_up": "final_models/xgb_regressor_bin_1000_up.pkl"
}
ATTRIBUTES_PATH = "utils/attributes.txt"

# === Pomocnicze funkcje
def haversine(lat1, lon1, lat2=41.3870, lon2=2.1701):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# === Wczytanie i przygotowanie danych
df = pd.read_csv(DATA_PATH, sep=";")
df = df[(df["price"] >= df["price"].quantile(0.01)) & (df["price"] <= df["price"].quantile(0.99))].copy()

# Feature engineering (tak samo jak przy trenowaniu!)
df["bathrooms_per_guest"] = df["bathrooms_num"] / df["accommodates"].replace(0, np.nan)
df["bedrooms_per_guest"] = df["bedrooms"] / df["accommodates"].replace(0, np.nan)
df["beds_per_guest"] = df["beds"] / df["accommodates"].replace(0, np.nan)
df["beds_per_bedroom"] = df["beds"] / df["bedrooms"].replace(0, np.nan)
df["guests_per_bedroom"] = df["accommodates"] / df["bedrooms"].replace(0, np.nan)
df["dist_to_center"] = df.apply(lambda row: haversine(row["latitude"], row["longitude"]), axis=1)

# Wczytaj KMeans i oblicz klaster lokalizacji
kmeans = joblib.load("final_models/kmeans_location_cluster.pkl")
df["location_cluster"] = kmeans.predict(df[["latitude", "longitude"]])

# Inne cechy
amenity_cols = [col for col in df.columns if col.startswith("amenity_")]
df["num_amenities"] = df[amenity_cols].sum(axis=1)
df["has_tv_or_wifi"] = df[["amenity_TV", "amenity_Wifi"]].max(axis=1)
df["is_suited_for_longterm"] = df[["amenity_Washer", "amenity_Kitchen", "amenity_Dishes and silverware"]].sum(axis=1) >= 2
df["lat_scaled"] = (df["latitude"] - df["latitude"].mean()) / df["latitude"].std()
df["lon_scaled"] = (df["longitude"] - df["longitude"].mean()) / df["longitude"].std()
df["is_group_friendly"] = (df["accommodates"] >= 4).astype(int)

# Naprawa inf i nan
for col in ["bathrooms_per_guest", "bedrooms_per_guest", "beds_per_guest", "beds_per_bedroom", "guests_per_bedroom"]:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    q_low, q_high = df[col].quantile([0.01, 0.99])
    df[col] = df[col].clip(q_low, q_high).fillna(df[col].median())

# === Wczytaj atrybuty i przygotuj X
with open(ATTRIBUTES_PATH) as f:
    feature_cols = [line.strip() for line in f]

X = df[feature_cols]
y_true = df["price"]
ids = df["id"]

# === Wczytanie modeli
classifier = joblib.load(CLASSIFIER_PATH)
regressors = {k: joblib.load(v) for k, v in REGRESSOR_PATHS.items()}

# === Predykcja
predicted_bins = classifier.predict(X)
predicted_prices = []

# Zakresy dla każdego binu
bin_ranges = {
    "bin_0_100": (0, 100),
    "bin_100_300": (100, 300),
    "bin_300_1000": (300, 1000),
    "bin_1000_up": (1000, np.inf)
}

for i in range(len(X)):
    bin_label = predicted_bins[i]
    regressor = regressors[bin_label]
    # Sprawdź, czy regresor był trenowany na log(price) — dotyczy binów powyżej 100 zł
    if bin_label == "bin_0_100":
        price_pred = regressor.predict(X.iloc[[i]])[0]
    else:
        price_pred = np.expm1(regressor.predict(X.iloc[[i]])[0])  # odwrotność log1p


    # Przycięcie do zakresu
    lower, upper = bin_ranges[bin_label]
    if price_pred < lower:
        price_pred = lower
    elif price_pred > upper:
        price_pred = upper
    elif price_pred < 50:
        price_pred = 50

    predicted_prices.append(price_pred)


# === Metryki
mae = mean_absolute_error(y_true, predicted_prices)
rmse = np.sqrt(mean_squared_error(y_true, predicted_prices))

r2 = r2_score(y_true, predicted_prices)

print("=== Global Metrics ===")
print(f"MAE:  {mae:.2f} zł")
print(f"RMSE: {rmse:.2f} zł")
print(f"R²:   {r2:.4f}")

# === Zapis wyników
# === Zapis wyników z dokładnością i błędem
bin_edges = [0, 100, 300, 1000, np.inf]
bin_labels = ["bin_0_100", "bin_100_300", "bin_300_1000", "bin_1000_up"]
true_bins = pd.cut(y_true, bins=bin_edges, labels=bin_labels)

correct = predicted_bins == true_bins.values
errors = np.abs(y_true - predicted_prices)

results = pd.DataFrame({
    "id": ids,
    "price": y_true,
    "predicted_bin": predicted_bins,
    "correct": correct,
    "predicted_price": predicted_prices,
    "error": errors
})

results.to_csv("evaluation_full_pipeline.csv", index=False)
print("Zapisano do evaluation_full_pipeline.csv")
