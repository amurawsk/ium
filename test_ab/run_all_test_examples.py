import pandas as pd
import requests
import json


def main(base_url, data_path, max_examples=10, verbose=False):
    df = pd.read_csv(data_path, sep=';')
    prices = df['price'].copy()

    df.drop(columns=['price'], inplace=True)

    records = df.to_dict(orient="records")

    counter = 0
    for i, record in enumerate(records, 1):
        if max_examples is not None and counter >= max_examples:
            break
        headers = {"Content-Type": "application/json"}
        if verbose:
            print(f'Actual price for row {i}: {prices[i-1]}')

        response_base = requests.post(f"{base_url}/predict_base", headers=headers, data=json.dumps(record))
        if verbose:
            print(f"Base prediction for row {i}: {response_base.json()}")

        response_adv = requests.post(f"{base_url}/predict_advanced", headers=headers, data=json.dumps(record))
        if verbose:
            print(f"Advanced prediction for row {i}: {response_adv.json()}")
        counter += 1

if __name__ == "__main__":
    main("http://localhost:8080", "../data/test_data.csv", None)
