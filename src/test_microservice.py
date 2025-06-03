import pickle
from flask import Flask, request, jsonify
import logging
import random
import numpy as np
import joblib
import logging
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from math import radians, sin, cos, sqrt, atan2

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)



BASE_MODEL_PATH = "final_models/baseline_model.pkl"
ADVANCED_MODEL_PATH = ""  # TODO
SCALER_PATH = ""  # TODO
ATTRIBUTES_NEEDED_INFO_PATH = "utils/attributes.txt"
LOGS_PATH = "../results/ab_test.log"
HOSTING_IP = "0.0.0.0"
HOSTING_PORT = 8080
CONST_SEED = 77


from sklearn.base import TransformerMixin, BaseEstimator

class PassthroughTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X



# wczytanie wcześniej wytrenowanych modeli
classifier = joblib.load("final_models/price_classifier.pkl")
regressors = {
    "bin_0_100": joblib.load("final_models/bin_0_100.pkl"),
    "bin_100_300": joblib.load("final_models/bin_100_300.pkl"),
    "bin_300_1000": joblib.load("final_models/bin_300_1000.pkl"),
    "bin_1000_up": joblib.load("final_models/bin_1000_up.pkl"),
}


# dodatkowa funkcja haversine (taka sama jak w treningu)
def haversine(lat1, lon1, lat2=41.3870, lon2=2.1701):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# przygotowanie dodatkowych cech, zgodnie z trenowaniem
import joblib

# Wczytaj model KMeans
kmeans = joblib.load("final_models/kmeans_location_cluster.pkl")

def prepare_features(input_data):
    df = pd.DataFrame([input_data])

    df["bathrooms_per_guest"] = df["bathrooms_num"] / df["accommodates"].replace(0, 1)
    df["bedrooms_per_guest"] = df["bedrooms"] / df["accommodates"].replace(0, 1)
    df["beds_per_guest"] = df["beds"] / df["accommodates"].replace(0, 1)
    df["beds_per_bedroom"] = df["beds"] / df["bedrooms"].replace(0, 1)
    df["guests_per_bedroom"] = df["accommodates"] / df["bedrooms"].replace(0, 1)
    df["dist_to_center"] = haversine(df["latitude"].iloc[0], df["longitude"].iloc[0])
    df["location_cluster"] = kmeans.predict(df[["latitude", "longitude"]])[0]  # <- to jest kluczowe
    df["num_amenities"] = df[[col for col in df if col.startswith("amenity_")]].sum(axis=1)
    df["has_tv_or_wifi"] = df[["amenity_TV", "amenity_Wifi"]].max(axis=1)
    longterm_features = ["amenity_Washer", "amenity_Kitchen", "amenity_Dishes and silverware"]
    df["is_suited_for_longterm"] = df[longterm_features].sum(axis=1) >= 2
    df["lat_scaled"] = (df["latitude"] - 41.3870) / 0.01
    df["lon_scaled"] = (df["longitude"] - 2.1701) / 0.01
    df["is_group_friendly"] = (df["accommodates"] >= 4).astype(int)

    return df





random.seed(CONST_SEED)
np.random.seed(CONST_SEED)

app = Flask(__name__)

logging.basicConfig(
    filename=LOGS_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

try:
    # with open(SCALER_PATH, "rb") as f:
    #     scaler = pickle.load(f)
    with open(BASE_MODEL_PATH, "rb") as f:
        median_price = pickle.load(f)
    # with open(ADVANCED_MODEL_PATH, "rb") as f:
    #     advanced_model = pickle.load(f)
except FileNotFoundError as e:
    logging.error(f"Nie udało się załadować modeli: {e}")
    raise RuntimeError("Nie udało się załadować modeli. Sprawdź pliki .pkl.")

with open(ATTRIBUTES_NEEDED_INFO_PATH, "r") as file:
    FEATURE_NAMES = [line.strip() for line in file.readlines()]


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Brak danych wejściowych."}), 400

        try:
            features = [data[feature] for feature in FEATURE_NAMES]
        except KeyError as e:
            missing_feature = e.args[0]
            return (
                jsonify({"error": f"Brakuje wymaganej cechy: {missing_feature}"}),
                400,
            )

        if random.random() < 0.5:
            model = 'basic'
            prediction = median_price
        else:
            model = 'advanced'
            # TODO use features list to get prediction from final model
            prediction = median_price

        logging.info(f"Zapytanie: {data}, Model: {model}, Prediction: {prediction}")
        return jsonify({"prediction": float(prediction)})
    except Exception as e:
        logging.error(f"Wystąpił błąd: {str(e)}")
        return jsonify({"error": "Wewnętrzny błąd serwera."}), 500


@app.route("/predict_base", methods=["POST"])
def predict_base():
    try:
        data = request.json
        base_prediction = median_price

        logging.info(f"Zapytanie: {data}, Base Prediction: {base_prediction}")
        return jsonify({"prediction": float(base_prediction)})
    except Exception as e:
        logging.error(f"Wystąpił błąd: {str(e)}")
        return jsonify({"error": "Wewnętrzny błąd serwera."}), 500


@app.route("/predict_advanced", methods=["POST"])
def predict_advanced():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Brak danych wejściowych."}), 400

        # upewnienie się, że wszystkie cechy są dostępne
        # try:
        #     features_input = {feature: data[feature] for feature in FEATURE_NAMES}
        # except KeyError as e:
        #     missing_feature = e.args[0]
        #     return jsonify({"error": f"Brakuje wymaganej cechy: {missing_feature}"}), 400

        features_input = data  # przyjmujemy tylko surowe dane od klienta


        # przygotowanie danych
        df_features = prepare_features(features_input)

        # klasyfikacja przedziału cenowego
        # bin_label = classifier.predict(df_features)[0]
        bin_label = "bin_100_300"  # dla uproszczenia, używamy tylko jednego binu

        print("=== DEBUG: Input to classifier ===")
        print(df_features.head(1).to_string())
        print("Shape:", df_features.shape)


        # regresja odpowiednim modelem
        model = regressors[bin_label]

        # predykcja ceny
        if bin_label == "bin_0_100":
            price_pred = model.predict(df_features)[0]  # bez log
        else:
            price_pred = np.expm1(model.predict(df_features)[0])  # exp dla log-cech

        logging.info(f"Zapytanie: {data}, Bin: {bin_label}, Prediction: {price_pred:.2f} zł")

        return jsonify({
            "predicted_price": round(price_pred, 2),
            "price_bin": bin_label
        })

    except Exception as e:
        logging.error(f"Wystąpił błąd: {str(e)}")
        return jsonify({"error": "Wewnętrzny błąd serwera."}), 500


if __name__ == "__main__":
    app.run(host=HOSTING_IP, port=HOSTING_PORT, debug=True)
