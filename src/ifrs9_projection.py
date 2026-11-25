import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# =====================================================
# 1. CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"

df = pd.read_csv(
    OUTPUT / "prepayment_predictions.csv"
)

metrics = pd.read_csv(
    OUTPUT / "prepayment_metrics.csv",
    index_col=0
)["value"]

partial_ratio = float(metrics["partial_ratio"])
total_rate = float(metrics["total_rate"])

HORIZON = 120

sns.set_theme(style="whitegrid")

# =====================================================
# 2. PARAMETRES FINANCIERS
# =====================================================

balance = df["outstanding_balance"].to_numpy()

rate = df["interest_rate"].to_numpy()

monthly_rate = rate / 12

remaining_months = (
    df["loan_term"] * 12 -
    df["loan_age_months"]
).to_numpy()

payment = df["monthly_payment"].to_numpy()

pd_annual = df["pd_estimated"].to_numpy()

prepay_annual = df["prepayment_conditional"].to_numpy()

lgd = np.where(
    df["guaranteed"].to_numpy() == 1,
    0.25,
    0.45
)

# Conversion des probabilites annuelles
# en probabilites mensuelles constantes

pd_monthly = 1 - (1 - pd_annual) ** (1 / 12)

# =====================================================
# 3. FONCTION DE PROJECTION
# =====================================================

def project_portfolio(name, multiplier):

    annual_prepay = np.clip(
        prepay_annual * multiplier,
        0,
        0.999
    )

    monthly_prepay = (
        1 - (1 - annual_prepay) ** (1 / 12)
    )

    # Capital contractuel par credit

    contractual = balance.copy()

    # Probabilite de rester actif sans defaut
    # ni remboursement total

    survival = np.ones(len(df))

    # Effet cumule des remboursements partiels

    partial_factor = np.ones(len(df))

    cumulative_ecl = 0.0

    rows = [{
        "scenario": name,
        "month": 0,
        "year": 0,
        "exposure": balance.sum(),
        "monthly_ecl": 0.0,
        "cumulative_ecl": 0.0
    }]

    for month in range(1, HORIZON + 1):

        active = remaining_months >= month

        # Exposition en debut de mois

        expected_exposure = (
            contractual *
            survival *
            partial_factor *
            active
        )

        # Pertes attendues actualisees
        # Le defaut intervient avant le remboursement
        # anticipe dans notre convention mensuelle

        discount = (1 + monthly_rate) ** month

        monthly_ecl = np.sum(
            expected_exposure *
            pd_monthly *
            lgd /
            discount
        )

        cumulative_ecl += monthly_ecl

        # Amortissement contractuel

        contractual = np.maximum(
            contractual * (1 + monthly_rate) -
            payment,
            0
        )

        contractual[remaining_months <= month] = 0

        # Sorties du portefeuille :
        # defauts et remboursements totaux

        p_total = monthly_prepay * total_rate

        p_partial = monthly_prepay * (1 - total_rate)

        survival *= (
            (1 - pd_monthly) *
            (1 - p_total)
        )

        # Remboursements partiels conditionnels
        # aux credits encore actifs

        conditional_partial = np.divide(
            p_partial,
            1 - p_total,
            out=np.zeros_like(p_partial),
            where=(1 - p_total) > 0
        )

        partial_factor *= (
            1 - conditional_partial * partial_ratio
        )

        exposure = np.sum(
            contractual *
            survival *
            partial_factor
        )

        rows.append({
            "scenario": name,
            "month": month,
            "year": month / 12,
            "exposure": exposure,
            "monthly_ecl": monthly_ecl,
            "cumulative_ecl": cumulative_ecl
        })

    return pd.DataFrame(rows)

# =====================================================
# 4. EXECUTION DES SCENARIOS
# =====================================================

scenarios = {
    "Contractuel": 0.0,
    "Comportemental": 1.0,
    "Remboursements faibles": 0.7,
    "Remboursements eleves": 1.3
}

results = pd.concat(
    [
        project_portfolio(name, multiplier)
        for name, multiplier in scenarios.items()
    ],
    ignore_index=True
)

results.to_csv(
    OUTPUT / "ead_ecl_projection.csv",
    index=False
)

# =====================================================
# 5. SYNTHESE DES RESULTATS
# =====================================================

summary_rows = []

for name in scenarios:

    scenario = results[
        results["scenario"] == name
    ]

    ecl_12m = scenario.loc[
        scenario["month"] == 12,
        "cumulative_ecl"
    ].iloc[0]

    ecl_10y = scenario.loc[
        scenario["month"] == 120,
        "cumulative_ecl"
    ].iloc[0]

    exposure_1y = scenario.loc[
        scenario["month"] == 12,
        "exposure"
    ].iloc[0]

    exposure_5y = scenario.loc[
        scenario["month"] == 60,
        "exposure"
    ].iloc[0]

    exposure_10y = scenario.loc[
        scenario["month"] == 120,
        "exposure"
    ].iloc[0]

    summary_rows.append({
        "scenario": name,
        "exposition_initiale": balance.sum(),
        "exposition_1_an": exposure_1y,
        "exposition_5_ans": exposure_5y,
        "exposition_10_ans": exposure_10y,
        "ecl_12_mois": ecl_12m,
        "ecl_10_ans": ecl_10y
    })

summary = pd.DataFrame(summary_rows)

baseline = summary.loc[
    summary["scenario"] == "Contractuel",
    "ecl_10_ans"
].iloc[0]

summary["reduction_ecl_10_ans_pct"] = (
    1 - summary["ecl_10_ans"] / baseline
) * 100

summary.to_csv(
    OUTPUT / "ifrs9_scenario_summary.csv",
    index=False
)

print("\n===== PORTEFEUILLE TEST =====")
print(f"Nombre de credits : {len(df)}")
print(f"Exposition initiale : {balance.sum():,.2f} EUR")
print(f"PD annuelle moyenne : {pd_annual.mean():.2%}")
print(f"LGD moyenne : {lgd.mean():.2%}")
print(f"Taux historique remboursement total : {total_rate:.2%}")
print(f"Ratio remboursement partiel : {partial_ratio:.2%}")

print("\n===== RESULTATS DES SCENARIOS =====")

for _, row in summary.iterrows():

    print("\nScenario :", row["scenario"])

    print(
        f"Exposition a 1 an : "
        f"{row['exposition_1_an']:,.2f} EUR"
    )

    print(
        f"Exposition a 5 ans : "
        f"{row['exposition_5_ans']:,.2f} EUR"
    )

    print(
        f"Exposition a 10 ans : "
        f"{row['exposition_10_ans']:,.2f} EUR"
    )

    print(
        f"ECL 12 mois : "
        f"{row['ecl_12_mois']:,.2f} EUR"
    )

    print(
        f"ECL cumulee 10 ans : "
        f"{row['ecl_10_ans']:,.2f} EUR"
    )

    print(
        f"Reduction ECL 10 ans : "
        f"{row['reduction_ecl_10_ans_pct']:.2f}%"
    )

# =====================================================
# 6. PROFILS D'ECOULEMENT
# =====================================================

plt.figure(figsize=(11, 6))

for name in scenarios:

    subset = results[
        results["scenario"] == name
    ]

    plt.plot(
        subset["year"],
        subset["exposure"] / 1e6,
        label=name,
        linewidth=2
    )

plt.title("Profils d'ecoulement du portefeuille immobilier")
plt.xlabel("Horizon de projection (annees)")
plt.ylabel("Exposition attendue (millions EUR)")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT / "11_exposure_runoff_profiles.png",
    dpi=200
)

plt.close()

# =====================================================
# 7. EVOLUTION DES PERTES ATTENDUES
# =====================================================

plt.figure(figsize=(11, 6))

for name in scenarios:

    subset = results[
        results["scenario"] == name
    ]

    plt.plot(
        subset["year"],
        subset["cumulative_ecl"] / 1e6,
        label=name,
        linewidth=2
    )

plt.title("Evolution des pertes de credit attendues")
plt.xlabel("Horizon de projection (annees)")
plt.ylabel("ECL cumulee actualisee (millions EUR)")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT / "12_cumulative_ecl.png",
    dpi=200
)

plt.close()

# =====================================================
# 8. COMPARAISON ECL 12 MOIS
# =====================================================

plt.figure(figsize=(10, 6))

plt.bar(
    summary["scenario"],
    summary["ecl_12_mois"] / 1e6
)

plt.title("Comparaison des ECL a 12 mois")
plt.ylabel("ECL (millions EUR)")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    OUTPUT / "13_ecl_12_months.png",
    dpi=200
)

plt.close()

# =====================================================
# 9. COMPARAISON ECL 10 ANS
# =====================================================

plt.figure(figsize=(10, 6))

plt.bar(
    summary["scenario"],
    summary["ecl_10_ans"] / 1e6
)

plt.title("Comparaison des ECL cumulees sur 10 ans")
plt.ylabel("ECL actualisee (millions EUR)")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    OUTPUT / "14_ecl_10_years.png",
    dpi=200
)

plt.close()

# =====================================================
# 10. CONTROLES DE COHERENCE
# =====================================================

assert (results["exposure"] >= 0).all()
assert (results["monthly_ecl"] >= 0).all()

for name in scenarios:

    subset = results[
        results["scenario"] == name
    ]

    assert (
        subset["exposure"].diff().dropna() <= 1e-6
    ).all()

    assert (
        subset["cumulative_ecl"].diff().dropna() >= -1e-6
    ).all()

assert (
    summary["ecl_10_ans"] <= baseline + 1e-6
).all()

print("\n===== CONTROLES DE COHERENCE =====")
print("Tous les controles sont valides.")

print("\n===== PROJECTION TERMINEE =====")
print("4 graphiques et 2 fichiers CSV generes.")
print("Prototype pedagogique IFRS 9.")
