from flask import Flask, request, jsonify
import logging
import random
import numpy as np
import joblib
import pandas as pd
from utils.custom_processing_classes import * # te klasy są potrzebne do modelu


MEDIAN_PRICE = 100
ADVANCED_MODEL_PATH = "final_models/best_model.joblib"
ATTRIBUTES_NEEDED_INFO_PATH = "utils/attributes.txt"
LOGS_PATH = "../results/microservice.log"
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
    advanced_model = joblib.load(ADVANCED_MODEL_PATH)
except Exception as e:
    logging.error(f"Nie udało się załadować modeli: {e}")
    raise RuntimeError("Nie udało się załadować modeli. Sprawdź pliki .pkl.")

with open(ATTRIBUTES_NEEDED_INFO_PATH, "r") as file:
    COLUMNS_TO_KEEP = [line.strip() for line in file.readlines()]


def validate_single_record(data_json):
    if not data_json:
        return False, "Brak danych wejściowych."
    if not isinstance(data_json, dict):
        return False, "Oczekiwany pojedynczy rekord JSON (słownik)."
    return True, None


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data_json = request.get_json()
        valid, error_msg = validate_single_record(data_json)
        if not valid:
            return jsonify({"error": error_msg}), 400

        if random.random() < 0.5:
            model_name = "basic"
            prediction = MEDIAN_PRICE
        else:
            model_name = "advanced"
            df = pd.DataFrame([data_json])
            df = df[COLUMNS_TO_KEEP]
            preds_log = advanced_model.predict(df)
            prediction = float(np.expm1(preds_log[0]))

        logging.info(f"Zapytanie: {data_json}, Model: {model_name}, Prediction: {prediction}")
        return jsonify({"prediction": prediction})
    except Exception as e:
        logging.error(f"Wystąpił błąd: {str(e)}")
        return jsonify({"error": "Wewnętrzny błąd serwera."}), 500


@app.route("/predict_base", methods=["POST"])
def predict_base():
    try:
        data_json = request.get_json()
        valid, error_msg = validate_single_record(data_json)
        if not valid:
            return jsonify({"error": error_msg}), 400

        base_prediction = MEDIAN_PRICE

        logging.info(f"Zapytanie: {data_json}, Base Prediction: {base_prediction}")
        return jsonify({"prediction": float(base_prediction)})
    except Exception as e:
        logging.error(f"Wystąpił błąd: {str(e)}")
        return jsonify({"error": "Wewnętrzny błąd serwera."}), 500


@app.route("/predict_advanced", methods=["POST"])
def predict_advanced():
    try:
        data_json = request.get_json()
        valid, error_msg = validate_single_record(data_json)
        if not valid:
            return jsonify({"error": error_msg}), 400

        df = pd.DataFrame([data_json])
        df = df[COLUMNS_TO_KEEP]
        preds_log = advanced_model.predict(df)
        prediction = float(np.expm1(preds_log[0]))

        logging.info(f"Zapytanie: {data_json}, Advanced Prediction: {prediction}")
        return jsonify({"prediction": prediction})
    except Exception as e:
        logging.error(f"Wystąpił błąd: {str(e)}")
        return jsonify({"error": "Wewnętrzny błąd serwera."}), 500


if __name__ == "__main__":
    app.run(host=HOSTING_IP, port=HOSTING_PORT, debug=True)
