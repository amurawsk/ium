import pickle
from flask import Flask, request, jsonify
import logging
import random
import numpy as np
import torch
import torch.nn as nn
from torch.serialization import add_safe_globals
from sklearn.preprocessing import StandardScaler

BASE_MODEL_PATH = "final_models/baseline_model.pkl"
ADVANCED_MODEL_PATH = "final_models/best_custom_model_all.pth"
SCALER_PATH = "final_models/best_custom_model_all.pth"  # scaler jest w tym samym pliku
ATTRIBUTES_NEEDED_INFO_PATH = "utils/attributes.txt"
LOGS_PATH = "../results/ab_test.log"
HOSTING_IP = "0.0.0.0"
HOSTING_PORT = 8080
CONST_SEED = 77

random.seed(CONST_SEED)
np.random.seed(CONST_SEED)

app = Flask(__name__)


class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.LeakyReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


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
    
    # Load the advanced model
    add_safe_globals([StandardScaler])
    checkpoint = torch.load(ADVANCED_MODEL_PATH, map_location='cpu', weights_only=False)
    input_dim = checkpoint['input_dim']
    advanced_model = RegressionModel(input_dim)
    advanced_model.load_state_dict(checkpoint['model_state_dict'])
    advanced_model.eval()
    scaler = checkpoint['scaler']


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
            return jsonify({"error": f"Brakuje cechy: {e.args[0]}"}), 400

        try:
            X = np.array(features).reshape(1, -1)
            X_scaled = scaler.transform(X)  # <-- tu często jest problem, jeśli kształt się nie zgadza
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            with torch.no_grad():
                advanced_prediction = advanced_model(X_tensor).item()
        except Exception as e:
            logging.error(f"Błąd podczas predykcji zaawansowanej: {str(e)}")
            return jsonify({"error": f"Błąd predykcji: {str(e)}"}), 500

        return jsonify({"prediction": float(advanced_prediction)})

    except Exception as e:
        logging.error(f"Wewnętrzny błąd: {str(e)}")
        return jsonify({"error": "Wewnętrzny błąd serwera."}), 500



if __name__ == "__main__":
    app.run(host=HOSTING_IP, port=HOSTING_PORT, debug=True)
