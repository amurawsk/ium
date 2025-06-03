import pandas as pd
import numpy as np
import os
import joblib
from math import radians, sin, cos, sqrt, atan2

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin

from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# === Ścieżki
DATA_PATH = "../data/final_data.csv"
MODEL_PATH = "final_models/price_classifier.pkl"
KMEANS_PATH = "final_models/kmeans_location_cluster.pkl"
ATTRIBUTES_PATH = "utils/attributes.txt"

os.makedirs("final_models", exist_ok=True)

# === Pomocnicza klasa
class PassthroughTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X

# === Funkcja odległości do centrum Barcelony
def haversine(lat1, lon1, lat2=41.3870, lon2=2.1701):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# === Wczytanie danych
df = pd.read_csv(DATA_PATH, sep=";")
df = df[(df["price"] >= df["price"].quantile(0.01)) & (df["price"] <= df["price"].quantile(0.99))].copy()

# === Feature engineering
df["bathrooms_per_guest"] = df["bathrooms_num"] / df["accommodates"].replace(0, np.nan)
df["bedrooms_per_guest"] = df["bedrooms"] / df["accommodates"].replace(0, np.nan)
df["beds_per_guest"] = df["beds"] / df["accommodates"].replace(0, np.nan)
df["beds_per_bedroom"] = df["beds"] / df["bedrooms"].replace(0, np.nan)
df["guests_per_bedroom"] = df["accommodates"] / df["bedrooms"].replace(0, np.nan)
df["dist_to_center"] = df.apply(lambda row: haversine(row["latitude"], row["longitude"]), axis=1)

# === Klasteryzacja lokalizacji
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=15, random_state=42)
df["location_cluster"] = kmeans.fit_predict(df[["latitude", "longitude"]])
joblib.dump(kmeans, KMEANS_PATH)

# === Dodatkowe cechy
amenity_cols = [col for col in df.columns if col.startswith("amenity_")]
df["num_amenities"] = df[amenity_cols].sum(axis=1)
df["has_tv_or_wifi"] = df[["amenity_TV", "amenity_Wifi"]].max(axis=1)
df["is_suited_for_longterm"] = df[["amenity_Washer", "amenity_Kitchen", "amenity_Dishes and silverware"]].sum(axis=1) >= 2
df["lat_scaled"] = (df["latitude"] - df["latitude"].mean()) / df["latitude"].std()
df["lon_scaled"] = (df["longitude"] - df["longitude"].mean()) / df["longitude"].std()
df["is_group_friendly"] = (df["accommodates"] >= 4).astype(int)

# === Czyszczenie cech
for col in ["bathrooms_per_guest", "bedrooms_per_guest", "beds_per_guest", "beds_per_bedroom", "guests_per_bedroom"]:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    q_low, q_high = df[col].quantile([0.01, 0.99])
    df[col] = df[col].clip(q_low, q_high).fillna(df[col].median())

# === Binning cen
bins = [0, 100, 300, 1000, np.inf]
labels = ["bin_0_100", "bin_100_300", "bin_300_1000", "bin_1000_up"]
df["price_bin"] = pd.cut(df["price"], bins=bins, labels=labels)

# === X, y
X = df.drop(columns=["price", "id", "price_bin"])
y = df["price_bin"]

# === Zapis listy wymaganych cech
with open(ATTRIBUTES_PATH, "w") as f:
    for col in X.columns:
        f.write(f"{col}\n")

# === Podział na train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# === Oversampling tylko na train
print("Rozkład klas przed SMOTE:")
print(y_train.value_counts(normalize=True))

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Rozkład klas po SMOTE:")
print(pd.Series(y_train_res).value_counts(normalize=True))

# === Pipeline
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

classifier_pipeline = Pipeline([
    ("pre", preprocessor),
    ("clf", HistGradientBoostingClassifier(random_state=42, class_weight="balanced"))
])

# === Trening
classifier_pipeline.fit(X_train_res, y_train_res)

# === Ewaluacja
y_pred = classifier_pipeline.predict(X_test)
print("=== Classification Report (test set) ===")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# === Macierz pomyłek
cm = confusion_matrix(y_test, y_pred, labels=labels)
disp = sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.savefig("final_models/confusion_matrix.png")
plt.close()

# === Zapis modelu
joblib.dump(classifier_pipeline, MODEL_PATH)
print(f"✅ Zapisano model klasyfikatora do {MODEL_PATH}")
