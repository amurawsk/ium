import pandas as pd
import re
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns

LOG_FILE = "../results/microservice.log"

pattern = re.compile(
    r"Zapytanie: (.+?), (Base|Advanced) Prediction: ([0-9.]+)"
)

data = []

with open(LOG_FILE, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            raw_json, model_type, prediction = match.groups()
            try:
                prediction = float(prediction)
                data.append({
                    "model": model_type.lower(),
                    "prediction": prediction
                })
            except ValueError:
                continue

df = pd.DataFrame(data)

print("\n== Liczba próbek per model ==")
print(df["model"].value_counts())
print("\n== Statystyki opisowe ==")
print(df.groupby("model")["prediction"].describe())

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x="prediction", hue="model", kde=True, bins=30, stat="density")
plt.title("Rozkład predykcji: base vs advanced")
plt.xlabel("Predykcja")
plt.ylabel("Gęstość")
plt.tight_layout()
plt.show()

base_preds = df[df["model"] == "base"]["prediction"]
adv_preds = df[df["model"] == "advanced"]["prediction"]

t_stat, p_val = ttest_ind(base_preds, adv_preds, equal_var=False)

print("\n== Test t-Studenta ==")
print(f"t = {t_stat:.4f}, p = {p_val:.4f}")
if p_val < 0.05:
    print("Różnica istotna statystycznie (p < 0.05)")
else:
    print("Brak istotnej różnicy (p ≥ 0.05)")
