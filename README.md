# Projekt IUM – Predykcja cen noclegów

Na początku warto zaznaczyć, że szczegółowe informacje dotyczące działania poszczególnych notebooków znajdują się bezpośrednio w ich treści – krok po kroku wyjaśniamy tam, jak działa preprocessing, modele oraz proces ewaluacji.

## Autorzy

* Adrian Murawski
* Jakub Wróblewski

---

## Uruchomienie mikroserwisu

Aby uruchomić mikroserwis, należy przejść do katalogu `microservice` i wykonać polecenie:

```bash
cd microservice
python3 microservice.py
```

Po uruchomieniu serwer będzie dostępny na porcie `8080`. Następnie można wysyłać zapytania typu POST. 
Po udanym zapytaniu serwer zwraca następującą wiadomość:

```json
{
  "prediction": 123.45
}
```

### Dostępne endpointy:
Aby poniższe komendy zadziałały poprawnie i ułatwić sobie życie, nalezy wejść do katalogu `utils` - tam znajduje się przygotowany plik z danymi wejściowymi.

```bash
cd utils
```

#### `/predict_base` – model bazowy

Zwraca zawsze tę samą wartość: 100 (mediana z danych treningowych). Nie korzysta z żadnego modelu ML.

```bash
curl -X POST http://localhost:8080/predict_base \
-H "Content-Type: application/json" \
-d @example_input.json
```

#### `/predict_advanced` – model zaawansowany

Używa wytrenowanego modelu Random Forest do predykcji na podstawie pełnych danych wejściowych. Wykorzystuje pipeline przetwarzania cech.

```bash
curl -X POST http://localhost:8080/predict_advanced \
-H "Content-Type: application/json" \
-d @example_input.json
```

#### `/predict` – wariant testowy A/B

Losowo wybiera jeden z dwóch modeli: bazowy lub zaawansowany (z prawdopodobieństwem 50/50). Służy do prowadzenia eksperymentów porównawczych.

```bash
curl -X POST http://localhost:8080/predict \
-H "Content-Type: application/json" \
-d @example_input.json

``` 

### Testowy input (`example_input.json`):
```json
{
  "id": "12345",
  "neighbourhood_cleansed": "Downtown",
  "neighbourhood_group_cleansed": "Central",
  "latitude": 41.3997,
  "longitude": 2.1575263,
  "property_type": "Apartment",
  "room_type": "Entire home/apt",
  "accommodates": 3,
  "bathrooms": 1.0,
  "bathrooms_text": "1 bath",
  "bedrooms": 1,
  "beds": 2,
  "amenities": "[\"Wifi\", \"Kitchen\", \"Heating\"]",
  "instant_bookable": "t",
  "price": 150
}
```
UWAGA: Pole `price` służy jedynie do porównania z wartością przewidywaną i oczywiście NIE jest wykorzystywane przez model.

---

## Metodologia i modele

### Analiza danych i inżynieria cech

W notebookach `notebooks/preprocess_data.ipynb` i `notebooks/data_analysis.ipynb` przeprowadziliśmy eksploracyjną analizę danych, na podstawie której opracowano zestaw cech do dalszego przetwarzania (feature engineering).

### Model bazowy

Notebook `notebooks/base_model.ipynb` zawiera bardzo prosty model, który dla każdego lokalu zwraca medianę cen z danych treningowych. Pełni rolę punktu odniesienia.

### Model finalny

W `notebooks/final_model.ipynb` znajduje się końcowy model oparty na Random Forest Regressor, który wykorzystuje zintegrowany pipeline przetwarzania cech oraz uczenia maszynowego.

### Starsze podejście (klasyfikacja + regresja)

Notebook `notebooks/old_classifier_regressors.ipynb` opisuje wcześniejsze podejście, w którym dane były najpierw klasyfikowane do jednego z przedziałów cenowych, a następnie wybierany był odpowiedni regresor do predykcji.

---

## Eksperyment A/B

W celu porównania skuteczności modeli zaimplementowano test A/B, który składa się z dwóch etapów:

### Wysyłanie danych testowych:

```bash
python test_ab/run_all_test_examples.py
```

Skrypt wysyła dane testowe do serwera i zapisuje odpowiedzi w `results/ab_test.log`.

### Analiza wyników:

```bash
python test_ab/analyze.py
```

Skrypt analizuje logi z pliku `results/microservice.log` i generuje podsumowanie.
