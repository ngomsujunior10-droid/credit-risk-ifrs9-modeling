import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

df = pd.read_csv(ROOT / "data/mortgage_portfolio.csv")

# 1. Controle de la qualite des donnees

print("\n===== QUALITE DES DONNEES =====")
print("Dimensions :", df.shape)
print("Valeurs manquantes :", df.isna().sum().sum())
print("Doublons :", df.duplicated().sum())
print("Identifiants uniques :", df["loan_id"].nunique())

assert df["loan_id"].is_unique
assert df.isna().sum().sum() == 0
assert (df["outstanding_balance"] >= 0).all()
assert (df["outstanding_balance"] <= df["loan_amount"]).all()
assert (df["prepayment_amount"] <= df["outstanding_balance"]).all()

print("Controles de qualite : OK")

# 2. Indicateurs du portefeuille

print("\n===== INDICATEURS DU PORTEFEUILLE =====")

summary = {
    "Nombre de credits": len(df),
    "Montant initial total": df["loan_amount"].sum(),
    "Exposition actuelle": df["outstanding_balance"].sum(),
    "Montant moyen du credit": df["loan_amount"].mean(),
    "Revenu annuel moyen": df["annual_income"].mean(),
    "Taux de defaut (%)": df["default"].mean() * 100,
    "Taux de remboursement anticipe (%)": df["prepayment"].mean() * 100,
    "Taux de credits cautionnes (%)": df["guaranteed"].mean() * 100,
    "Montant rembourse par anticipation": df["prepayment_amount"].sum()
}

for key, value in summary.items():
    print(f"{key} : {value:,.2f}")

pd.Series(summary).to_csv(
    OUTPUT / "portfolio_summary.csv",
    header=["value"]
)

# 3. Analyse par niveau d'endettement

df["dti_group"] = pd.cut(
    df["dti"],
    bins=[0, 0.25, 0.35, 0.45, np.inf],
    labels=["<25%", "25-35%", "35-45%", ">45%"],
    include_lowest=True
)

default_by_dti = (
    df.groupby("dti_group", observed=True)["default"].mean() * 100
)

print("\n===== TAUX DE DEFAUT PAR DTI =====")
print(default_by_dti.round(2))

# 4. Analyse par caution

guarantee_analysis = df.groupby("guaranteed").agg(
    nombre_credits=("loan_id", "count"),
    exposition=("outstanding_balance", "sum"),
    taux_defaut=("default", "mean"),
    taux_remboursement_anticipe=("prepayment", "mean")
)

guarantee_analysis["taux_defaut"] *= 100
guarantee_analysis["taux_remboursement_anticipe"] *= 100

print("\n===== ANALYSE PAR CAUTION =====")
print(guarantee_analysis.round(2).to_string())

guarantee_analysis.to_csv(
    OUTPUT / "guarantee_analysis.csv"
)

# 5. Distribution des credits

plt.figure(figsize=(9, 5))
sns.histplot(df["loan_amount"], bins=40, kde=True)
plt.title("Distribution des credits immobiliers")
plt.xlabel("Montant du credit (EUR)")
plt.ylabel("Nombre de credits")
plt.tight_layout()
plt.savefig(OUTPUT / "01_loan_distribution.png", dpi=200)
plt.close()

# 6. Risque de defaut et endettement

plt.figure(figsize=(8, 5))
default_by_dti.plot(kind="bar")
plt.title("Taux de defaut selon le taux d'endettement")
plt.xlabel("Taux d'endettement (DTI)")
plt.ylabel("Taux de defaut (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT / "02_default_by_dti.png", dpi=200)
plt.close()

# 7. Remboursements anticipes

plt.figure(figsize=(7, 7))

counts = df["prepayment_type"].value_counts().reindex(
    ["none", "partial", "total"],
    fill_value=0
)

plt.pie(
    counts,
    labels=["Aucun", "Partiel", "Total"],
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Repartition des remboursements anticipes")
plt.tight_layout()
plt.savefig(OUTPUT / "03_prepayment_distribution.png", dpi=200)
plt.close()

# 8. Score de credit et defaut

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="default", y="credit_score")
plt.title("Score de credit et risque de defaut")
plt.xlabel("Statut de defaut")
plt.ylabel("Score de credit")
plt.xticks([0, 1], ["Non defaillant", "Defaillant"])
plt.tight_layout()
plt.savefig(OUTPUT / "04_credit_score_default.png", dpi=200)
plt.close()

# 9. Matrice de correlation

features = [
    "annual_income",
    "loan_amount",
    "interest_rate",
    "ltv",
    "dti",
    "credit_score",
    "outstanding_balance",
    "default",
    "prepayment"
]

plt.figure(figsize=(11, 8))

sns.heatmap(
    df[features].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Matrice de correlation des variables financieres")
plt.tight_layout()
plt.savefig(OUTPUT / "05_correlation_matrix.png", dpi=200)
plt.close()

print("\n===== ANALYSE TERMINEE =====")
print("5 graphiques et 2 fichiers CSV generes.")
print("Dossier :", OUTPUT)
