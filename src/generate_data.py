
import numpy as np
import pandas as pd
from pathlib import Path

# =====================================================
# 1. CONFIGURATION
# =====================================================

rng = np.random.default_rng(42)
N = 10_000

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# =====================================================
# 2. CARACTERISTIQUES DES EMPRUNTEURS
# =====================================================

age = rng.integers(23, 71, N)

annual_income = np.clip(
    rng.lognormal(np.log(42000), 0.45, N),
    18000, 200000
)

credit_score = np.clip(
    rng.normal(680, 65, N),
    300, 850
).astype(int)

# =====================================================
# 3. CARACTERISTIQUES DES CREDITS IMMOBILIERS
# =====================================================

property_value = np.clip(
    annual_income * rng.uniform(3, 7, N),
    80000, 900000
)

ltv = rng.uniform(0.50, 0.95, N)

loan_amount = property_value * ltv

interest_rate = rng.uniform(0.015, 0.055, N)

loan_term = rng.choice(
    [10, 15, 20, 25, 30],
    size=N,
    p=[0.05, 0.15, 0.35, 0.35, 0.10]
)

guaranteed = rng.choice(
    [0, 1],
    size=N,
    p=[0.30, 0.70]
)

# =====================================================
# 4. MENSUALITES ET TAUX D'ENDETTEMENT
# =====================================================

monthly_rate = interest_rate / 12
n_months = loan_term * 12

monthly_payment = (
    loan_amount * monthly_rate
    / (1 - (1 + monthly_rate) ** (-n_months))
)

dti = monthly_payment / (annual_income / 12)

# =====================================================
# 5. CAPITAL RESTANT DU
# =====================================================

loan_age_months = (
    rng.random(N) * (n_months - 12)
).astype(int)

outstanding_balance = (
    loan_amount * (1 + monthly_rate) ** loan_age_months
    - monthly_payment
    * ((1 + monthly_rate) ** loan_age_months - 1)
    / monthly_rate
)

outstanding_balance = np.maximum(
    outstanding_balance, 0
)

# =====================================================
# 6. SIMULATION DU RISQUE DE DEFAUT
# =====================================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


default_logit = (
    -3.8
    + 5 * (dti - 0.30)
    + 3 * (ltv - 0.70)
    - 0.012 * (credit_score - 680)
    + 10 * (interest_rate - 0.03)
)

pd_true = sigmoid(default_logit)

default = rng.binomial(1, pd_true)

# =====================================================
# 7. REMBOURSEMENTS ANTICIPES
# =====================================================

market_rate = 0.032

prepayment_logit = (
    -2.2
    + 35 * (interest_rate - market_rate)
    + 0.000015 * (annual_income - 42000)
    - 2 * (dti - 0.30)
    + 0.003 * (credit_score - 680)
)

prepayment_probability = sigmoid(prepayment_logit)

prepayment = rng.binomial(
    1, prepayment_probability
)

# Hypothese simplificatrice :
# defaut et remboursement anticipe exclusifs

prepayment[default == 1] = 0

prepayment_type = np.full(
    N, "none", dtype=object
)

mask = prepayment == 1

prepayment_type[mask] = rng.choice(
    ["partial", "total"],
    size=mask.sum(),
    p=[0.65, 0.35]
)

prepayment_amount = np.zeros(N)

partial_mask = prepayment_type == "partial"
total_mask = prepayment_type == "total"

prepayment_amount[partial_mask] = (
    outstanding_balance[partial_mask]
    * rng.uniform(0.10, 0.40, partial_mask.sum())
)

prepayment_amount[total_mask] = (
    outstanding_balance[total_mask]
)

# =====================================================
# 8. CREATION DU DATASET
# =====================================================

df = pd.DataFrame({
    "loan_id": np.arange(1, N + 1),
    "age": age,
    "annual_income": annual_income.round(2),
    "loan_amount": loan_amount.round(2),
    "interest_rate": interest_rate.round(4),
    "loan_term": loan_term,
    "property_value": property_value.round(2),
    "ltv": ltv.round(4),
    "monthly_payment": monthly_payment.round(2),
    "dti": dti.round(4),
    "credit_score": credit_score,
    "loan_age_months": loan_age_months,
    "outstanding_balance": outstanding_balance.round(2),
    "guaranteed": guaranteed,
    "default": default,
    "prepayment": prepayment,
    "prepayment_type": prepayment_type,
    "prepayment_amount": prepayment_amount.round(2)
})

# =====================================================
# 9. EXPORTATION DES DONNEES
# =====================================================

output_file = DATA_DIR / "mortgage_portfolio.csv"

df.to_csv(output_file, index=False)

# =====================================================
# 10. RESUME DU PORTEFEUILLE
# =====================================================

print("=" * 60)
print("PORTEFEUILLE DE CREDITS IMMOBILIERS")
print("=" * 60)

print(f"Dimensions : {df.shape}")
print(f"Nombre de credits : {len(df):,}")

print(
    f"Montant total initial : "
    f"{df['loan_amount'].sum():,.2f} EUR"
)

print(
    f"Capital restant du total : "
    f"{df['outstanding_balance'].sum():,.2f} EUR"
)

print(
    f"Taux de defaut : "
    f"{df['default'].mean():.2%}"
)

print(
    f"Taux de remboursement anticipe : "
    f"{df['prepayment'].mean():.2%}"
)

print("\nTypes de remboursement anticipe :")
print(df["prepayment_type"].value_counts())

print("\nValeurs manquantes :")
print(df.isna().sum().sum())

print("\nApercu des donnees :")
print(df.head())

print(f"\nFichier genere : {output_file}")
