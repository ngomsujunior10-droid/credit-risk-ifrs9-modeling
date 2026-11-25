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
    roc_auc_score,
    roc_curve,
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

# 2. Variables explicatives

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

# Meme separation que pour le modele PD

train_idx, test_idx = train_test_split(
    df.index,
    test_size=0.20,
    random_state=42,
    stratify=df["default"]
)

train = df.loc[train_idx].copy()
test = df.loc[test_idx].copy()

# 3. Fonction de construction du modele

def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])

# 4. Modele de remboursement anticipe
# Conditionnel a l'absence de defaut

train_surv = train[train["default"] == 0]
test_surv = test[test["default"] == 0]

model = build_model()

model.fit(
    train_surv[features],
    train_surv["prepayment"]
)

p_test = model.predict_proba(
    test_surv[features]
)[:, 1]

y_test = test_surv["prepayment"]

auc = roc_auc_score(y_test, p_test)
brier = brier_score_loss(y_test, p_test)
ap = average_precision_score(y_test, p_test)

baseline_brier = brier_score_loss(
    y_test,
    np.full(len(y_test), train_surv["prepayment"].mean())
)

print("\n===== MODELE DE REMBOURSEMENT ANTICIPE =====")

print(f"ROC AUC : {auc:.4f}")
print(f"Gini : {2 * auc - 1:.4f}")
print(f"Brier Score : {brier:.4f}")
print(f"Brier baseline : {baseline_brier:.4f}")
print(f"Average Precision : {ap:.4f}")
print(f"Probabilite moyenne estimee : {p_test.mean():.2%}")
print(f"Taux observe conditionnel : {y_test.mean():.2%}")

# 5. Courbe ROC

fpr, tpr, _ = roc_curve(y_test, p_test)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], "--", label="Aleatoire")
plt.xlabel("Taux de faux positifs")
plt.ylabel("Taux de vrais positifs")
plt.title("ROC - Remboursement anticipe")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT / "09_prepayment_roc.png", dpi=200)
plt.close()

# 6. Coefficients du modele

coef = pd.DataFrame({
    "variable": features,
    "coefficient": model.named_steps["logistic"].coef_[0]
}).sort_values("coefficient")

print("\n===== FACTEURS EXPLICATIFS =====")
print(coef.round(4).to_string(index=False))

coef.to_csv(
    OUTPUT / "prepayment_coefficients.csv",
    index=False
)

plt.figure(figsize=(9, 6))
plt.barh(coef["variable"], coef["coefficient"])
plt.axvline(0, linestyle="--")
plt.title("Facteurs du remboursement anticipe")
plt.xlabel("Coefficient standardise")
plt.tight_layout()
plt.savefig(OUTPUT / "10_prepayment_coefficients.png", dpi=200)
plt.close()

# 7. Modele conditionnel de remboursement total

train_prepay = train[train["prepayment"] == 1].copy()
test_prepay = test[test["prepayment"] == 1].copy()

y_total_train = (
    train_prepay["prepayment_type"] == "total"
).astype(int)

y_total_test = (
    test_prepay["prepayment_type"] == "total"
).astype(int)

total_model = build_model()

total_model.fit(
    train_prepay[features],
    y_total_train
)

p_total_test = total_model.predict_proba(
    test_prepay[features]
)[:, 1]

total_auc = roc_auc_score(
    y_total_test,
    p_total_test
)

total_brier = brier_score_loss(
    y_total_test,
    p_total_test
)

total_baseline_brier = brier_score_loss(
    y_total_test,
    np.full(len(y_total_test), y_total_train.mean())
)

print("\n===== MODELE DE REMBOURSEMENT TOTAL =====")
print(f"ROC AUC : {total_auc:.4f}")
print(f"Brier Score : {total_brier:.4f}")
print(f"Brier baseline : {total_baseline_brier:.4f}")
print(f"Probabilite moyenne : {p_total_test.mean():.2%}")
print(f"Taux observe : {y_total_test.mean():.2%}")

# Choix prudent : taux historique si le modele
# n'ameliore pas le Brier Score de reference

total_rate = y_total_train.mean()

use_total_model = total_brier < total_baseline_brier

print("Modele retenu :", (
    "Regression logistique"
    if use_total_model
    else "Taux historique"
))

# 8. Estimation du remboursement partiel

partial = train[
    (train["prepayment_type"] == "partial") &
    (train["outstanding_balance"] > 0)
].copy()

partial_ratio = (
    partial["prepayment_amount"] /
    partial["outstanding_balance"]
).mean()

print("\n===== REMBOURSEMENT PARTIEL =====")
print(f"Part moyenne du capital rembourse : {partial_ratio:.2%}")

# 9. Predictions sur le portefeuille test

pd_model = joblib.load(
    MODELS / "pd_logistic_model.joblib"
)

result = test[[
    "loan_id",
    "outstanding_balance",
    "guaranteed",
    "interest_rate",
    "loan_term",
    "loan_age_months",
    "monthly_payment"
]].copy()

result["pd_estimated"] = pd_model.predict_proba(
    test[features]
)[:, 1]

result["prepayment_conditional"] = model.predict_proba(
    test[features]
)[:, 1]

result["prepayment_probability"] = (
    (1 - result["pd_estimated"]) *
    result["prepayment_conditional"]
)

if use_total_model:
    result["total_conditional"] = total_model.predict_proba(
        test[features]
    )[:, 1]
else:
    result["total_conditional"] = total_rate

result["partial_probability"] = (
    result["prepayment_probability"] *
    (1 - result["total_conditional"])
)

result["total_probability"] = (
    result["prepayment_probability"] *
    result["total_conditional"]
)

result["expected_prepayment"] = (
    result["outstanding_balance"] *
    (
        result["total_probability"] +
        result["partial_probability"] * partial_ratio
    )
)

assert (
    result["pd_estimated"] +
    result["prepayment_probability"] <= 1.000001
).all()

# 10. Exportation des resultats

result.to_csv(
    OUTPUT / "prepayment_predictions.csv",
    index=False
)

metrics = {
    "roc_auc": auc,
    "gini": 2 * auc - 1,
    "brier_score": brier,
    "baseline_brier": baseline_brier,
    "average_precision": ap,
    "total_auc": total_auc,
    "total_brier": total_brier,
    "total_baseline_brier": total_baseline_brier,
    "total_rate": total_rate,
    "partial_ratio": partial_ratio
}

pd.Series(metrics).to_csv(
    OUTPUT / "prepayment_metrics.csv",
    header=["value"]
)

# 11. Sauvegarde des modeles

joblib.dump(
    model,
    MODELS / "prepayment_model.joblib"
)

joblib.dump({
    "model": total_model,
    "use_model": use_total_model,
    "total_rate": total_rate,
    "partial_ratio": partial_ratio
}, MODELS / "prepayment_total_model.joblib")

print("\n===== SYNTHESE SUR LE PORTEFEUILLE TEST =====")

print(f"Nombre de credits : {len(result)}")

print(
    f"Probabilite moyenne de remboursement anticipe : "
    f"{result['prepayment_probability'].mean():.2%}"
)

print(
    f"Montant attendu des remboursements anticipes : "
    f"{result['expected_prepayment'].sum():,.2f} EUR"
)

print("\n===== MODELISATION TERMINEE =====")
print("Modeles et resultats sauvegardes.")
