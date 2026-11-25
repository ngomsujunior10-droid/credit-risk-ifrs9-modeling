import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    brier_score_loss,
    average_precision_score
)

# 1. Configuration

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
MODELS = ROOT / "models"

OUTPUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "data/mortgage_portfolio.csv")

sns.set_theme(style="whitegrid")

# 2. Selection des variables explicatives

features = [
    "age",
    "annual_income",
    "loan_amount",
    "interest_rate",
    "loan_term",
    "ltv",
    "dti",
    "credit_score",
    "loan_age_months",
    "outstanding_balance",
    "guaranteed"
]

X = df[features]
y = df["default"]

# 3. Separation Train / Test

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n===== SEPARATION DES DONNEES =====")
print("Train :", X_train.shape)
print("Test :", X_test.shape)
print(f"Taux de defaut train : {y_train.mean():.2%}")
print(f"Taux de defaut test : {y_test.mean():.2%}")

# 4. Construction du modele

model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

model.fit(X_train, y_train)

# 5. Prediction

y_proba = model.predict_proba(X_test)[:, 1]

threshold = 0.50

y_pred = (y_proba >= threshold).astype(int)

# 6. Evaluation

auc = roc_auc_score(y_test, y_proba)
accuracy = accuracy_score(y_test, y_pred)
brier = brier_score_loss(y_test, y_proba)
gini = 2 * auc - 1
ap = average_precision_score(y_test, y_proba)

baseline = np.full(len(y_test), y_train.mean())
baseline_brier = brier_score_loss(y_test, baseline)

print("\n===== PERFORMANCE DU MODELE PD =====")

print(f"ROC AUC : {auc:.4f}")
print(f"Accuracy : {accuracy:.2%}")
print(f"Coefficient de Gini : {gini:.4f}")
print(f"Average Precision : {ap:.4f}")
print(f"Brier Score : {brier:.4f}")
print(f"Brier Score baseline : {baseline_brier:.4f}")
print(f"PD moyenne estimee : {y_proba.mean():.2%}")
print(f"Taux de defaut observe : {y_test.mean():.2%}")

print("\n===== RAPPORT DE CLASSIFICATION =====")

print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

# 7. Matrice de confusion

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Non-defaut", "Defaut"],
    yticklabels=["Non-defaut", "Defaut"]
)

plt.title("Matrice de confusion - Modele PD")
plt.xlabel("Prediction")
plt.ylabel("Observation")
plt.tight_layout()
plt.savefig(OUTPUT / "06_pd_confusion_matrix.png", dpi=200)
plt.close()

# 8. Courbe ROC

fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Regression logistique - AUC = {auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Modele aleatoire"
)

plt.xlabel("Taux de faux positifs")
plt.ylabel("Taux de vrais positifs")
plt.title("Courbe ROC - Probabilite de defaut")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT / "07_pd_roc_curve.png", dpi=200)
plt.close()

# 9. Analyse des coefficients

coefficients = pd.DataFrame({
    "variable": features,
    "coefficient": model.named_steps["logistic"].coef_[0]
})

coefficients = coefficients.sort_values("coefficient")

print("\n===== COEFFICIENTS DU MODELE =====")
print(coefficients.round(4).to_string(index=False))

coefficients.to_csv(
    OUTPUT / "pd_coefficients.csv",
    index=False
)

plt.figure(figsize=(9, 6))

plt.barh(
    coefficients["variable"],
    coefficients["coefficient"]
)

plt.axvline(0, linestyle="--")
plt.title("Coefficients standardises du modele PD")
plt.xlabel("Coefficient")
plt.tight_layout()
plt.savefig(OUTPUT / "08_pd_coefficients.png", dpi=200)
plt.close()

# 10. Exportation des predictions hors entrainement

predictions = df.loc[X_test.index, [
    "loan_id",
    "default",
    "outstanding_balance",
    "guaranteed"
]].copy()

predictions["pd_estimated"] = y_proba

predictions.to_csv(
    OUTPUT / "pd_predictions.csv",
    index=False
)

# 11. Exportation des performances

metrics = {
    "roc_auc": auc,
    "accuracy": accuracy,
    "gini": gini,
    "average_precision": ap,
    "brier_score": brier,
    "baseline_brier": baseline_brier,
    "mean_predicted_pd": y_proba.mean(),
    "observed_default_rate": y_test.mean()
}

pd.Series(metrics).to_csv(
    OUTPUT / "pd_metrics.csv",
    header=["value"]
)

# 12. Sauvegarde du modele

joblib.dump(
    model,
    MODELS / "pd_logistic_model.joblib"
)

print("\n===== MODELISATION PD TERMINEE =====")
print("Modele sauvegarde avec succes.")
print("3 graphiques et 3 fichiers CSV generes.")
