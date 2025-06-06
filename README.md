# Projekt IUM
## Autorzy
- Adrian Murawski
- Jakub Wróblewski

## Opis problemu
Opracowanie modelu predykcyjnego, który na podstawie cech lokalu (np. lokalizacja, metraż, typ, liczba pokoi, dostępność udogodnień) oszacuje optymalną cenę za noc dla danego lokalu, porównywalną do cen podobnych lokali dostępnych na rynku.

## Opis rozwiązania
Zanalizowano dostępne dane - proces analizy dostępny jest w `notebooks/data_analysis.ipynb`. Następnie stworzono pipeline do modelu, który pozwala na przetworzenie danych zgodnie z tym co ustalono w procesie analizy. 

Stworzono bazowy model (najprostszy dla zdefiniowanego zadania) - serwujący medianę cen lokali w zbiorze treningowym jako predykcję (niezależnie od danych wejściowych).

Jako finalny model wybrano RandomForestRegressor. Model generuje predykcje ceny lokalu na podstawie dostarczonych cech (przeprocesowanych w pipeline'ie). Cały proces treningu, ewaluacja modelu oraz porównanie z modelem bazowym dostępne są w pliku `notebooks/final_model.ipynb`.

## Uruchomienie mikroserwisu
Mikroserwis uruchamia się w bardzo prosty sposób. Należy, będąc w katalogu `microservice` uruchomić polecenie `python microservice.py` (uprzednio instalując wymagane pakiety).

## Wygenerowanie logu predykcji - realizacja eksperymentu A/B
Logi do późniejszej ewaluacji można wygenerować przy pomocy skryptu `test_ab/run_all_test_examples.py`, który wszystkie elementy ze zbioru testowego wysyła do serwera, który następnie generuje predykcje.

## Ewaluacja testu A/B na podstawie logu
Na podstawie stworzonego logu można dokonać ewaluacji - służy do tego skrypt `test_ab/analyze.py`.
