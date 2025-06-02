import pickle
from flask import Flask, request, jsonify
import logging
import random
import numpy as np

BASE_MODEL_PATH = "final_models/baseline_model.pkl"
ADVANCED_MODEL_PATH = ""  # TODO
SCALER_PATH = ""  # TODO
ATTRIBUTES_NEEDED_INFO_PATH = "utils/attributes.txt"
LOGS_PATH = "../results/ab_test.log"
HOSTING_IP = "0.0.0.0"
HOSTING_PORT = 8080
CONST_SEED = 77

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

        try:
            features = [data[feature] for feature in FEATURE_NAMES]
        except KeyError as e:
            missing_feature = e.args[0]
            return (jsonify({"error": f"Brakuje wymaganej cechy: {missing_feature}"}), 400)

        # TODO use features list to get prediction from final model
        advanced_prediction = median_price

        logging.info(f"Zapytanie: {data}, Advanced Prediction: {advanced_prediction}")
        return jsonify({"prediction": float(advanced_prediction)})
    except Exception as e:
        logging.error(f"Wystąpił błąd: {str(e)}")
        return jsonify({"error": "Wewnętrzny błąd serwera."}), 500


if __name__ == "__main__":
    app.run(host=HOSTING_IP, port=HOSTING_PORT, debug=True)
