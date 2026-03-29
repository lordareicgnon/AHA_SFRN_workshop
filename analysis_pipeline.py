"""
SFRN Analysis Pipeline
======================
Stress–Cardiovascular Pathways: Integrating Survey, Geospatial, and Proteomic Data

Step 1: Data Preparation
Step 2: Clustering of Survey Data (Latent Stress Phenotypes)
Step 3: Mapping Clusters to Geospatial Exposures
Step 4: Differential Protein Expression Across Clusters
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import folium
from folium.plugins import HeatMap
from branca.colormap import LinearColormap
import requests
import networkx as nx

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

PSYCHOSOCIAL_TOTALS = [
    "PHQ8_total",    # depression
    "GAD2_total",    # anxiety (brief)
    "PSS10_total",   # perceived stress
    "PCL6_total",    # PTSD symptoms
    "PROMIS_anx_raw",# PROMIS anxiety
    "PROMIS_dep_raw",# PROMIS depression
    "PSQI_global",   # sleep quality
]

CLUSTER_LABELS_MAP = {0: "Low Burden", 1: "Moderate Burden", 2: "High Burden"}

# Protein markers grouped by function
PROTEIN_COLS = [
    "P02741_CRP", "P05231_IL6", "P01375_TNF_alpha",        # inflammatory
    "P16860_NT_proBNP", "P19429_Troponin_I",                # cardiac stress
    "P15692_VEGF_A", "P14780_MMP9", "P05362_ICAM1",         # vascular
    "P05121_PAI1", "P17931_Galectin3", "Q01638_ST2_IL1RL1", # fibrosis/thrombosis
    "Q99988_GDF15", "P05305_Endothelin1",                    # misc stress
    "Q15848_Adiponectin",                                    # protective
    "Q13884_Lp_PLA2",                                        # vascular inflammation
]

PROTEIN_SHORT = {
    "P02741_CRP": "CRP",
    "P05231_IL6": "IL-6",
    "P01375_TNF_alpha": "TNF-α",
    "P16860_NT_proBNP": "NT-proBNP",
    "P19429_Troponin_I": "Troponin I",
    "P15692_VEGF_A": "VEGF-A",
    "P14780_MMP9": "MMP-9",
    "P05362_ICAM1": "ICAM-1",
    "P05121_PAI1": "PAI-1",
    "P17931_Galectin3": "Galectin-3",
    "Q01638_ST2_IL1RL1": "ST2/IL1RL1",
    "Q99988_GDF15": "GDF-15",
    "P05305_Endothelin1": "Endothelin-1",
    "Q15848_Adiponectin": "Adiponectin",
    "Q13884_Lp_PLA2": "Lp-PLA2",
}


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — Data Preparation
# ════════════════════════════════════════════════════════════════════════════
def step1_data_preparation():
    print("=" * 72)
    print("STEP 1 — DATA PREPARATION")
    print("=" * 72)

    survey = pd.read_csv(DATA_DIR / "survey_data.csv")
    geo = pd.read_csv(DATA_DIR / "geospatial_data.csv")
    omics = pd.read_csv(DATA_DIR / "omics_data.csv")

    print(f"\nSurvey:      {survey.shape[0]:,} participants × {survey.shape[1]} columns")
    print(f"Geospatial:  {geo.shape[0]:,} ZIP codes × {geo.shape[1]} columns")
    print(f"Omics:       {omics.shape[0]:,} participants × {omics.shape[1]} columns")

    # ── Missingness ──────────────────────────────────────────────────────
    print("\n── Missingness ──")
    for name, df in [("Survey", survey), ("Geospatial", geo), ("Omics", omics)]:
        missing = df.isnull().sum()
        total_missing = missing.sum()
        if total_missing == 0:
            print(f"  {name}: no missing values")
        else:
            print(f"  {name}: {total_missing} missing values")
            print(missing[missing > 0].to_string(header=False))

    # ── Summary statistics for psychosocial instruments ──────────────────
    print("\n── Psychosocial Instrument Summary ──")
    print(survey[PSYCHOSOCIAL_TOTALS].describe().round(2).to_string())

    # ── Distribution plots ───────────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for i, col in enumerate(PSYCHOSOCIAL_TOTALS):
        axes[i].hist(survey[col], bins=20, edgecolor="black", alpha=0.7, color="steelblue")
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel("Score")
        axes[i].set_ylabel("Count")
    axes[-1].axis("off")
    plt.suptitle("Distributions of Psychosocial Instrument Scores", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "step1_psychosocial_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Correlation heatmap ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = survey[PSYCHOSOCIAL_TOTALS].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, square=True)
    ax.set_title("Correlation Among Psychosocial Instruments")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "step1_psychosocial_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Demographics summary ─────────────────────────────────────────────
    print("\n── Demographics ──")
    print(f"  Age: mean={survey['age'].mean():.1f}, std={survey['age'].std():.1f}")
    print(f"  Sex: {survey['sex'].value_counts().to_dict()}")
    print(f"  Race/ethnicity: {survey['race_ethnicity'].value_counts().to_dict()}")
    print(f"  Unique ZIP codes: {survey['zip_code'].nunique()}")

    print(f"\n  Saved: step1_psychosocial_distributions.png")
    print(f"  Saved: step1_psychosocial_correlation.png")

    return survey, geo, omics


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — Clustering of Survey Data
# ════════════════════════════════════════════════════════════════════════════
def step2_clustering(survey):
    print("\n" + "=" * 72)
    print("STEP 2 — CLUSTERING OF SURVEY DATA (Latent Stress Phenotypes)")
    print("=" * 72)

    X = survey[PSYCHOSOCIAL_TOTALS].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Model selection: compare K-Means and GMM across k=2..6 ───────────
    K_range = range(2, 7)
    results = {"k": [], "kmeans_silhouette": [], "gmm_silhouette": [], "gmm_bic": []}

    for k in K_range:
        # K-Means
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        km_labels = km.fit_predict(X_scaled)
        km_sil = silhouette_score(X_scaled, km_labels)

        # GMM
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                               n_init=5, random_state=42)
        gmm_labels = gmm.fit_predict(X_scaled)
        gmm_sil = silhouette_score(X_scaled, gmm_labels)
        gmm_bic = gmm.bic(X_scaled)

        results["k"].append(k)
        results["kmeans_silhouette"].append(km_sil)
        results["gmm_silhouette"].append(gmm_sil)
        results["gmm_bic"].append(gmm_bic)

    results_df = pd.DataFrame(results)
    print("\n── Cluster Selection Metrics ──")
    print(results_df.to_string(index=False))

    # ── Plot model selection ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(results_df["k"], results_df["kmeans_silhouette"], "o-", label="K-Means")
    axes[0].plot(results_df["k"], results_df["gmm_silhouette"], "s-", label="GMM")
    axes[0].set_xlabel("Number of Clusters")
    axes[0].set_ylabel("Silhouette Score")
    axes[0].set_title("Silhouette Score by k")
    axes[0].legend()
    axes[0].set_xticks(list(K_range))

    axes[1].plot(results_df["k"], results_df["gmm_bic"], "o-", color="green")
    axes[1].set_xlabel("Number of Clusters")
    axes[1].set_ylabel("BIC")
    axes[1].set_title("GMM BIC by k")
    axes[1].set_xticks(list(K_range))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "step2_cluster_selection.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fit final model with k=3 ────────────────────────────────────────
    optimal_k = 3
    print(f"\n  Selected k = {optimal_k}")

    gmm_final = GaussianMixture(n_components=optimal_k, covariance_type="full",
                                 n_init=10, random_state=42)
    cluster_labels = gmm_final.fit_predict(X_scaled)

    # Order clusters by mean total burden (sum of z-scores) so that
    # 0 = Low, 1 = Moderate, 2 = High
    burden = pd.DataFrame(X_scaled, columns=PSYCHOSOCIAL_TOTALS)
    burden["cluster_raw"] = cluster_labels
    mean_burden = burden.groupby("cluster_raw").mean().sum(axis=1).sort_values()
    rank_map = {old: new for new, old in enumerate(mean_burden.index)}
    cluster_labels_ordered = np.array([rank_map[c] for c in cluster_labels])

    survey = survey.copy()
    survey["stress_cluster"] = cluster_labels_ordered
    survey["stress_phenotype"] = survey["stress_cluster"].map(CLUSTER_LABELS_MAP)

    # ── Cluster sizes ────────────────────────────────────────────────────
    print("\n── Cluster Sizes ──")
    for label in sorted(CLUSTER_LABELS_MAP.keys()):
        n = (survey["stress_cluster"] == label).sum()
        print(f"  Cluster {label} ({CLUSTER_LABELS_MAP[label]}): n={n}")

    # ── Cluster profiles ─────────────────────────────────────────────────
    print("\n── Mean Psychosocial Scores by Cluster ──")
    profile = survey.groupby("stress_phenotype")[PSYCHOSOCIAL_TOTALS].mean().round(2)
    # Reorder rows
    profile = profile.reindex(["Low Burden", "Moderate Burden", "High Burden"])
    print(profile.to_string())

    # ── Radar / parallel‐coordinates plot ────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"Low Burden": "#2ecc71", "Moderate Burden": "#f39c12", "High Burden": "#e74c3c"}
    for pheno in ["Low Burden", "Moderate Burden", "High Burden"]:
        means = profile.loc[pheno]
        ax.plot(means.index, means.values, "o-", label=pheno, color=colors[pheno], linewidth=2)
    ax.set_xticklabels(PSYCHOSOCIAL_TOTALS, rotation=45, ha="right")
    ax.set_ylabel("Mean Score")
    ax.set_title("Psychosocial Instrument Profiles by Stress Phenotype")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "step2_cluster_profiles.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── ANOVA for each instrument across clusters ────────────────────────
    print("\n── ANOVA: Instrument Scores by Cluster ──")
    for col in PSYCHOSOCIAL_TOTALS:
        groups = [g[col].values for _, g in survey.groupby("stress_cluster")]
        f_stat, p_val = stats.f_oneway(*groups)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"  {col:20s}  F={f_stat:8.2f}  p={p_val:.2e}  {sig}")

    print(f"\n  Saved: step2_cluster_selection.png")
    print(f"  Saved: step2_cluster_profiles.png")

    return survey


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — Mapping Clusters to Geospatial Exposures
# ════════════════════════════════════════════════════════════════════════════
def step3_geospatial(survey, geo):
    print("\n" + "=" * 72)
    print("STEP 3 — MAPPING CLUSTERS TO GEOSPATIAL EXPOSURES")
    print("=" * 72)

    # ── Join on zip_code ─────────────────────────────────────────────────
    merged = survey.merge(geo, on="zip_code", how="left")
    unmatched = merged[geo.columns.drop("zip_code")].isnull().any(axis=1).sum()
    print(f"\n  Merged survey + geospatial: {merged.shape[0]} rows, {unmatched} unmatched ZIPs")

    geo_features = [c for c in geo.columns if c != "zip_code"]

    # ── Mean exposure by cluster ─────────────────────────────────────────
    print("\n── Mean Geospatial Exposures by Stress Phenotype ──")
    geo_profile = merged.groupby("stress_phenotype")[geo_features].mean().round(3)
    geo_profile = geo_profile.reindex(["Low Burden", "Moderate Burden", "High Burden"])
    print(geo_profile.T.to_string())

    # ── Statistical tests (Kruskal-Wallis for non-normal geo data) ───────
    print("\n── Kruskal-Wallis Tests: Exposures by Cluster ──")
    sig_features = []
    for col in geo_features:
        groups = [g[col].dropna().values for _, g in merged.groupby("stress_cluster")]
        h_stat, p_val = stats.kruskal(*groups)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"  {col:30s}  H={h_stat:8.2f}  p={p_val:.2e}  {sig}")
        if p_val < 0.05:
            sig_features.append(col)

    # ── Post-hoc pairwise comparisons for significant features ───────────
    if sig_features:
        print("\n── Post-hoc Tukey HSD for Significant Features ──")
        for col in sig_features[:5]:  # limit output
            tukey = pairwise_tukeyhsd(merged[col], merged["stress_phenotype"])
            print(f"\n  {col}:")
            print("  " + str(tukey).replace("\n", "\n  "))

    # ── Heatmap of standardized exposures ────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    geo_z = geo_profile.apply(stats.zscore, axis=0)
    sns.heatmap(geo_z.T, annot=geo_profile.T.values, fmt=".2f", cmap="YlOrRd",
                ax=ax, linewidths=0.5, cbar_kws={"label": "Z-score"})
    ax.set_title("Geospatial Exposure Profiles by Stress Phenotype\n(color = z-score, annotations = raw means)")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "step3_geospatial_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Boxplots for key exposures ───────────────────────────────────────
    key_exposures = ["ADI_national_rank", "PM25_annual_mean", "violent_crime_per100k",
                     "food_desert_score", "NDVI_greenspace", "SVI_overall",
                     "median_household_income", "noise_dBA_Ldn"]
    available = [c for c in key_exposures if c in merged.columns]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    order = ["Low Burden", "Moderate Burden", "High Burden"]
    palette = {"Low Burden": "#2ecc71", "Moderate Burden": "#f39c12", "High Burden": "#e74c3c"}
    for i, col in enumerate(available):
        sns.boxplot(data=merged, x="stress_phenotype", y=col, order=order,
                    palette=palette, ax=axes[i])
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel("")
    for j in range(len(available), len(axes)):
        axes[j].axis("off")
    plt.suptitle("Key Geospatial Exposures by Stress Phenotype", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "step3_exposure_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  Saved: step3_geospatial_heatmap.png")
    print(f"  Saved: step3_exposure_boxplots.png")

    return merged


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — Differential Protein Expression Across Clusters
# ════════════════════════════════════════════════════════════════════════════
def step4_proteomics(survey, omics):
    print("\n" + "=" * 72)
    print("STEP 4 — DIFFERENTIAL PROTEIN EXPRESSION ACROSS CLUSTERS")
    print("=" * 72)

    # ── Join on participant_id ───────────────────────────────────────────
    merged = survey[["participant_id", "stress_cluster", "stress_phenotype"]].merge(
        omics, on="participant_id", how="inner"
    )
    print(f"\n  Merged: {merged.shape[0]} participants with both survey + omics data")

    available_proteins = [c for c in PROTEIN_COLS if c in merged.columns]

    # ── Mean expression by cluster ───────────────────────────────────────
    print("\n── Mean Protein Expression by Stress Phenotype ──")
    protein_profile = merged.groupby("stress_phenotype")[available_proteins].mean().round(3)
    protein_profile = protein_profile.reindex(["Low Burden", "Moderate Burden", "High Burden"])
    display_cols = {c: PROTEIN_SHORT.get(c, c) for c in available_proteins}
    print(protein_profile.rename(columns=display_cols).T.to_string())

    # ── ANOVA / Kruskal-Wallis for each protein ─────────────────────────
    print("\n── ANOVA: Protein Expression by Cluster ──")
    anova_results = []
    for col in available_proteins:
        groups = [g[col].dropna().values for _, g in merged.groupby("stress_cluster")]
        f_stat, p_val = stats.f_oneway(*groups)
        short = PROTEIN_SHORT.get(col, col)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        anova_results.append({"Protein": short, "F": f_stat, "p": p_val, "sig": sig})
        print(f"  {short:15s}  F={f_stat:8.2f}  p={p_val:.2e}  {sig}")

    # ── Post-hoc Tukey for significant proteins ──────────────────────────
    sig_proteins = [r for r in anova_results if r["p"] < 0.05]
    if sig_proteins:
        print(f"\n── Post-hoc Tukey HSD ({len(sig_proteins)} significant proteins) ──")
        for r in sig_proteins:
            col = [c for c, s in PROTEIN_SHORT.items() if s == r["Protein"]][0]
            tukey = pairwise_tukeyhsd(merged[col], merged["stress_phenotype"])
            print(f"\n  {r['Protein']}:")
            print("  " + str(tukey).replace("\n", "\n  "))

    # ── Effect sizes (Cohen's d: High vs Low) ───────────────────────────
    print("\n── Effect Sizes (Cohen's d: High Burden vs Low Burden) ──")
    low = merged[merged["stress_cluster"] == 0]
    high = merged[merged["stress_cluster"] == 2]
    for col in available_proteins:
        m1, s1 = low[col].mean(), low[col].std()
        m2, s2 = high[col].mean(), high[col].std()
        pooled_std = np.sqrt((s1**2 + s2**2) / 2)
        d = (m2 - m1) / pooled_std if pooled_std > 0 else 0
        short = PROTEIN_SHORT.get(col, col)
        direction = "↑" if d > 0 else "↓"
        magnitude = "large" if abs(d) >= 0.8 else "medium" if abs(d) >= 0.5 else "small"
        print(f"  {short:15s}  d={d:+.3f}  {direction} ({magnitude})")

    # ── Heatmap of protein expression ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    profile_z = protein_profile.apply(stats.zscore, axis=0)
    annot_labels = protein_profile.rename(columns=display_cols)
    heat_data = profile_z.rename(columns=display_cols)
    sns.heatmap(heat_data.T, annot=annot_labels.T.values, fmt=".3f", cmap="RdBu_r",
                center=0, ax=ax, linewidths=0.5, cbar_kws={"label": "Z-score"})
    ax.set_title("Protein Expression by Stress Phenotype\n(color = z-score, annotations = raw mean log₂ abundance)")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "step4_protein_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Boxplots for key proteins ────────────────────────────────────────
    key_proteins = ["P02741_CRP", "P05231_IL6", "P01375_TNF_alpha",
                    "P16860_NT_proBNP", "P19429_Troponin_I", "Q15848_Adiponectin",
                    "P05362_ICAM1", "Q99988_GDF15"]
    available_key = [c for c in key_proteins if c in merged.columns]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    order = ["Low Burden", "Moderate Burden", "High Burden"]
    palette = {"Low Burden": "#2ecc71", "Moderate Burden": "#f39c12", "High Burden": "#e74c3c"}
    for i, col in enumerate(available_key):
        sns.boxplot(data=merged, x="stress_phenotype", y=col, order=order,
                    palette=palette, ax=axes[i])
        axes[i].set_title(PROTEIN_SHORT.get(col, col), fontsize=11)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("log₂ abundance")
    for j in range(len(available_key), len(axes)):
        axes[j].axis("off")
    plt.suptitle("Key Protein Markers by Stress Phenotype", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "step4_protein_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Volcano-style summary ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in anova_results:
        col = [c for c, s in PROTEIN_SHORT.items() if s == r["Protein"]][0]
        m_low = merged.loc[merged["stress_cluster"] == 0, col].mean()
        m_high = merged.loc[merged["stress_cluster"] == 2, col].mean()
        log2fc = m_high - m_low  # already in log space
        neg_log_p = -np.log10(r["p"]) if r["p"] > 0 else 30
        color = "#e74c3c" if r["p"] < 0.05 and log2fc > 0 else \
                "#2ecc71" if r["p"] < 0.05 and log2fc < 0 else "gray"
        ax.scatter(log2fc, neg_log_p, c=color, s=80, edgecolors="black", linewidth=0.5)
        ax.annotate(r["Protein"], (log2fc, neg_log_p), fontsize=8,
                    xytext=(5, 5), textcoords="offset points")
    ax.axhline(-np.log10(0.05), color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Log₂ Fold Change (High Burden vs Low Burden)")
    ax.set_ylabel("-log₁₀(p-value)")
    ax.set_title("Differential Protein Expression: High vs Low Stress Burden")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "step4_volcano_plot.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  Saved: step4_protein_heatmap.png")
    print(f"  Saved: step4_protein_boxplots.png")
    print(f"  Saved: step4_volcano_plot.png")

    return merged


# ════════════════════════════════════════════════════════════════════════════
# CV DISEASE PREVALENCE BY CLUSTER
# ════════════════════════════════════════════════════════════════════════════
def plot_cv_prevalence(survey):
    print("\n" + "=" * 72)
    print("CV DISEASE PREVALENCE BY STRESS PHENOTYPE")
    print("=" * 72)

    dx_cols = {
        "dx_hypertension": "Hypertension",
        "dx_diabetes": "Diabetes",
        "dx_heart_failure": "Heart Failure",
        "dx_MI_history": "MI History",
        "dx_stroke_history": "Stroke History",
    }
    available = [c for c in dx_cols if c in survey.columns]

    order = ["Low Burden", "Moderate Burden", "High Burden"]
    palette = {"Low Burden": "#2ecc71", "Moderate Burden": "#f39c12", "High Burden": "#e74c3c"}

    # Compute prevalence (%) and counts per cluster
    prev_data = []
    for pheno in order:
        sub = survey[survey["stress_phenotype"] == pheno]
        n = len(sub)
        for col in available:
            count = sub[col].sum()
            pct = count / n * 100
            prev_data.append({"Stress Phenotype": pheno, "Condition": dx_cols[col],
                              "Prevalence (%)": pct, "n": count, "N": n})
    prev_df = pd.DataFrame(prev_data)

    # Print table
    print("\n── Prevalence (%) ──")
    pivot = prev_df.pivot(index="Condition", columns="Stress Phenotype", values="Prevalence (%)")
    pivot = pivot[order].round(1)
    print(pivot.to_string())

    # Chi-square tests
    print("\n── Chi-Square Tests ──")
    for col in available:
        ct = pd.crosstab(survey["stress_phenotype"], survey[col])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {dx_cols[col]:20s}  χ²={chi2:8.2f}  p={p:.2e}  {sig}")

    # ── Grouped bar chart ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    conditions = [dx_cols[c] for c in available]
    x = np.arange(len(conditions))
    width = 0.25

    for i, pheno in enumerate(order):
        vals = prev_df[prev_df["Stress Phenotype"] == pheno].set_index("Condition").loc[conditions, "Prevalence (%)"]
        bars = ax.bar(x + i * width, vals, width, label=pheno, color=palette[pheno], edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.set_ylabel("Prevalence (%)")
    ax.set_title("Cardiovascular Disease Prevalence by Stress Phenotype")
    ax.legend()
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cv_disease_prevalence.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  Saved: cv_disease_prevalence.png")


# ════════════════════════════════════════════════════════════════════════════
# GEOSPATIAL HEATMAPS
# ════════════════════════════════════════════════════════════════════════════
DX_COLS = {
    "dx_hypertension": "Hypertension",
    "dx_diabetes": "Diabetes",
    "dx_heart_failure": "Heart Failure",
    "dx_MI_history": "MI History",
    "dx_stroke_history": "Stroke History",
}

# Embedded ZIP-code centroids (Sacramento region) — no external dependency
ZIP_COORDS = {
    95608: (38.6284, -121.3287), 95610: (38.6946, -121.2692),
    95621: (38.6952, -121.3075), 95624: (38.4232, -121.3599),
    95630: (38.6709, -121.1529), 95660: (38.6707, -121.3781),
    95670: (38.6072, -121.2761), 95673: (38.6895, -121.4479),
    95757: (38.4081, -121.4294), 95758: (38.4243, -121.4370),
    95762: (38.6850, -121.0680), 95811: (38.5762, -121.4880),
    95814: (38.5804, -121.4922), 95815: (38.6093, -121.4443),
    95816: (38.5728, -121.4675), 95817: (38.5498, -121.4583),
    95818: (38.5568, -121.4929), 95819: (38.5683, -121.4366),
    95820: (38.5347, -121.4451), 95821: (38.6239, -121.3837),
    95822: (38.5091, -121.4935), 95823: (38.4797, -121.4438),
    95824: (38.5178, -121.4419), 95825: (38.5892, -121.4057),
    95826: (38.5539, -121.3693), 95827: (38.5662, -121.3286),
    95828: (38.4826, -121.4006), 95829: (38.4689, -121.3440),
    95830: (38.4896, -121.2772), 95831: (38.4962, -121.5297),
    95832: (38.4695, -121.4883), 95833: (38.6157, -121.5053),
    95834: (38.6383, -121.5072), 95835: (38.6626, -121.4834),
    95836: (38.7198, -121.5343), 95837: (38.6817, -121.6030),
    95838: (38.6406, -121.4440), 95841: (38.6627, -121.3406),
    95842: (38.6865, -121.3494), 95843: (38.7159, -121.3648),
    95864: (38.5878, -121.3769),
}


def _get_zip_coords(zip_codes):
    """Return a DataFrame with zip_code, latitude, longitude from embedded coords."""
    records = []
    for z in zip_codes:
        if z in ZIP_COORDS:
            lat, lon = ZIP_COORDS[z]
            records.append({"zip_code": z, "latitude": lat, "longitude": lon})
    return pd.DataFrame(records)


def _make_heatmap(df, value_col, title, gradient=None):
    """Create a folium density heatmap weighted by value_col, with
    a street-map base layer and labeled ZIP markers underneath."""
    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11,
                   tiles="OpenStreetMap")

    # Add additional tile layers the user can toggle
    folium.TileLayer("CartoDB positron", name="Light").add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark").add_to(m)

    # Normalize values to 0-1 for heatmap weighting
    vmin, vmax = df[value_col].min(), df[value_col].max()
    rng = vmax - vmin if vmax > vmin else 1.0

    # Build weighted heat data: repeat each point proportionally to its value
    # so folium HeatMap renders a proper density surface
    heat_data = []
    for _, row in df.iterrows():
        weight = (row[value_col] - vmin) / rng  # 0-1
        heat_data.append([row["latitude"], row["longitude"], weight])

    if gradient is None:
        gradient = {0.2: "#ffffb2", 0.4: "#fecc5c", 0.6: "#fd8d3c",
                    0.8: "#f03b20", 1.0: "#bd0026"}

    HeatMap(
        heat_data,
        min_opacity=0.35,
        max_zoom=13,
        radius=35,
        blur=25,
        gradient=gradient,
    ).add_to(m)

    # Add small ZIP label markers on top
    for _, row in df.iterrows():
        val = row[value_col]
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=4,
            color="black", fill=True, fill_color="white", fill_opacity=0.8,
            weight=1,
            popup=f"<b>ZIP {int(row['zip_code'])}</b><br>{title}: {val:.2f}",
            tooltip=f"{int(row['zip_code'])}: {val:.2f}",
        ).add_to(m)

    # Add a title banner
    title_html = f'''
    <div style="position:fixed; top:10px; left:60px; z-index:1000;
         background:rgba(255,255,255,0.85); padding:8px 16px;
         border-radius:6px; font-size:16px; font-weight:bold;
         box-shadow:0 2px 6px rgba(0,0,0,0.3);">
        {title}
    </div>'''
    m.get_root().html.add_child(folium.Element(title_html))

    folium.LayerControl().add_to(m)
    return m


def generate_geospatial_heatmaps(survey, geo):
    """Generate interactive folium density heatmaps for psychosocial,
    environmental, and disease burden patterns at the ZIP-code level."""
    print("\n" + "=" * 72)
    print("GEOSPATIAL HEATMAPS")
    print("=" * 72)

    # Get ZIP coordinates
    all_zips = sorted(geo["zip_code"].unique())
    coords = _get_zip_coords(all_zips)
    print(f"\n  Resolved coordinates for {len(coords)}/{len(all_zips)} ZIP codes")

    # ── ZIP-level aggregations from survey ───────────────────────────────
    zip_psych = survey.groupby("zip_code")[PSYCHOSOCIAL_TOTALS].mean()
    scaler = StandardScaler()
    z_scores = pd.DataFrame(
        scaler.fit_transform(zip_psych), index=zip_psych.index, columns=PSYCHOSOCIAL_TOTALS
    )
    zip_psych["psych_burden_composite"] = z_scores.mean(axis=1)
    zip_psych = zip_psych.reset_index()

    # Disease prevalence by ZIP
    zip_dx = survey.groupby("zip_code")[list(DX_COLS.keys())].mean() * 100
    zip_dx["cv_disease_burden"] = zip_dx.mean(axis=1)
    zip_dx = zip_dx.reset_index()

    # Participant counts
    zip_n = survey.groupby("zip_code").size().reset_index(name="n_participants")

    # ── Merge all ZIP-level data with coordinates ────────────────────────
    zip_all = (coords.merge(geo, on="zip_code")
                     .merge(zip_psych, on="zip_code")
                     .merge(zip_dx, on="zip_code")
                     .merge(zip_n, on="zip_code"))

    # Color gradients
    hot_gradient = {0.2: "#ffffb2", 0.4: "#fecc5c", 0.6: "#fd8d3c",
                    0.8: "#f03b20", 1.0: "#bd0026"}
    cool_gradient = {0.2: "#f7fcf5", 0.4: "#c7e9c0", 0.6: "#74c476",
                     0.8: "#238b45", 1.0: "#00441b"}
    purple_gradient = {0.2: "#f2f0f7", 0.4: "#cbc9e2", 0.6: "#9e9ac8",
                       0.8: "#756bb1", 1.0: "#54278f"}

    # ── 1. Psychosocial Burden Heatmaps ──────────────────────────────────
    print("\n  Generating psychosocial burden maps...")
    m = _make_heatmap(zip_all, "psych_burden_composite",
                      "Psychosocial Burden (composite z-score)", purple_gradient)
    m.save(str(OUTPUT_DIR / "map_psychosocial_burden.html"))
    print(f"  Saved: map_psychosocial_burden.html")

    for col in PSYCHOSOCIAL_TOTALS:
        label = col.replace("_", " ").title()
        m = _make_heatmap(zip_all, col, f"Mean {label}", purple_gradient)
        m.save(str(OUTPUT_DIR / f"map_{col}.html"))
    print(f"  Saved: map_<instrument>.html (7 individual instrument maps)")

    # ── 2. Environmental Exposure Heatmaps ───────────────────────────────
    print("\n  Generating environmental exposure maps...")
    env_features = {
        "ADI_national_rank": ("Area Deprivation Index", hot_gradient),
        "PM25_annual_mean": ("PM2.5 Annual Mean (µg/m³)", hot_gradient),
        "violent_crime_per100k": ("Violent Crime per 100k", hot_gradient),
        "food_desert_score": ("Food Desert Score", hot_gradient),
        "NDVI_greenspace": ("Green Space (NDVI)", cool_gradient),
        "SVI_overall": ("Social Vulnerability Index", hot_gradient),
        "median_household_income": ("Median Household Income ($)", cool_gradient),
        "noise_dBA_Ldn": ("Noise Level (dBA Ldn)", hot_gradient),
    }
    for col, (label, grad) in env_features.items():
        if col in zip_all.columns:
            m = _make_heatmap(zip_all, col, label, grad)
            m.save(str(OUTPUT_DIR / f"map_{col}.html"))
    print(f"  Saved: map_<exposure>.html (8 environmental maps)")

    # ── 3. Disease Burden Heatmaps ───────────────────────────────────────
    print("\n  Generating disease burden maps...")
    m = _make_heatmap(zip_all, "cv_disease_burden",
                      "CV Disease Burden (mean prevalence %)", hot_gradient)
    m.save(str(OUTPUT_DIR / "map_cv_disease_burden.html"))
    print(f"  Saved: map_cv_disease_burden.html")

    for col, label in DX_COLS.items():
        if col in zip_all.columns:
            m = _make_heatmap(zip_all, col, f"{label} Prevalence (%)", hot_gradient)
            m.save(str(OUTPUT_DIR / f"map_{col}.html"))
    print(f"  Saved: map_dx_<condition>.html (5 disease maps)")

    # ── 4. Combined multi-layer heatmap ──────────────────────────────────
    print("\n  Generating combined multi-layer map...")
    center_lat = zip_all["latitude"].mean()
    center_lon = zip_all["longitude"].mean()
    m_combined = folium.Map(location=[center_lat, center_lon], zoom_start=11,
                            tiles="OpenStreetMap")
    folium.TileLayer("CartoDB positron", name="Light").add_to(m_combined)
    folium.TileLayer("CartoDB dark_matter", name="Dark").add_to(m_combined)

    layers = {
        "Psychosocial Burden": ("psych_burden_composite", purple_gradient),
        "Area Deprivation (ADI)": ("ADI_national_rank", hot_gradient),
        "PM2.5 Pollution": ("PM25_annual_mean", hot_gradient),
        "CV Disease Burden": ("cv_disease_burden", hot_gradient),
        "Green Space (NDVI)": ("NDVI_greenspace", cool_gradient),
        "Violent Crime": ("violent_crime_per100k", hot_gradient),
    }

    for layer_name, (col, grad) in layers.items():
        fg = folium.FeatureGroup(name=layer_name,
                                 show=(layer_name == "Psychosocial Burden"))
        vmin, vmax = zip_all[col].min(), zip_all[col].max()
        rng = vmax - vmin if vmax > vmin else 1.0
        heat_data = []
        for _, row in zip_all.iterrows():
            w = (row[col] - vmin) / rng
            heat_data.append([row["latitude"], row["longitude"], w])
        HeatMap(heat_data, min_opacity=0.3, radius=35, blur=25,
                gradient=grad).add_to(fg)
        fg.add_to(m_combined)

    # ZIP labels on top
    marker_fg = folium.FeatureGroup(name="ZIP Labels", show=True)
    for _, row in zip_all.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=3, color="black", fill=True, fill_color="white",
            fill_opacity=0.8, weight=1,
            tooltip=f"ZIP {int(row['zip_code'])}",
        ).add_to(marker_fg)
    marker_fg.add_to(m_combined)

    folium.LayerControl(collapsed=False).add_to(m_combined)
    m_combined.save(str(OUTPUT_DIR / "map_combined_layers.html"))
    print(f"  Saved: map_combined_layers.html (toggle layers with checkboxes)")

    # ── Summary tile plot ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plot_data = [
        ("psych_burden_composite", "Psychosocial Burden", "YlOrRd"),
        ("ADI_national_rank", "Area Deprivation Index", "YlOrRd"),
        ("PM25_annual_mean", "PM2.5 (µg/m³)", "YlOrRd"),
        ("cv_disease_burden", "CV Disease Burden (%)", "YlOrRd"),
        ("NDVI_greenspace", "Green Space (NDVI)", "YlGn"),
        ("violent_crime_per100k", "Violent Crime /100k", "YlOrRd"),
    ]
    for ax, (col, title, cmap_name) in zip(axes.flatten(), plot_data):
        sc = ax.scatter(zip_all["longitude"], zip_all["latitude"],
                        c=zip_all[col], cmap=cmap_name, s=120,
                        edgecolors="black", linewidth=0.5, alpha=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.colorbar(sc, ax=ax, shrink=0.8)
    plt.suptitle("Geospatial Burden Patterns by ZIP Code (Sacramento Region)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "geospatial_burden_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: geospatial_burden_summary.png")

    return zip_all


# ════════════════════════════════════════════════════════════════════════════
# TISSUE EXPRESSION (Human Protein Atlas)
# ════════════════════════════════════════════════════════════════════════════
UNIPROT_TO_GENE = {
    "P02741": "CRP", "P05231": "IL6", "P01375": "TNF",
    "P16860": "NPPB", "P19429": "TNNI3", "P15692": "VEGFA",
    "P14780": "MMP9", "P05362": "ICAM1", "P05121": "SERPINE1",
    "P17931": "LGALS3", "Q01638": "IL1RL1", "Q99988": "GDF15",
    "P05305": "EDN1", "Q15848": "ADIPOQ", "Q13884": "PLA2G7",
}

HPA_TISSUES = [
    "heart", "liver", "kidney", "lung", "brain", "adipose+tissue",
    "bone+marrow", "spleen", "colon", "small+intestine", "stomach",
    "pancreas", "skin", "smooth+muscle", "lymph+node",
    "adrenal+gland", "thyroid+gland", "esophagus", "gallbladder",
    "urinary+bladder", "placenta", "breast", "endometrium",
    "ovary", "prostate", "testis", "salivary+gland",
]


def analyze_tissue_expression(survey, omics):
    """Fetch tissue expression from HPA and visualize organ-level patterns."""
    print("\n" + "=" * 72)
    print("TISSUE EXPRESSION (Human Protein Atlas)")
    print("=" * 72)

    # Fetch from HPA
    print("\n  Fetching RNA expression from proteinatlas.org...")
    tissue_cols_query = ",".join([f"t_RNA_{t}" for t in HPA_TISSUES])
    records = []
    for uid, gene in UNIPROT_TO_GENE.items():
        url = (f"https://www.proteinatlas.org/api/search_download.php"
               f"?search={uid}&format=json&columns=g,up,{tissue_cols_query}&compress=no")
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200 and r.json():
                entry = r.json()[0]
                row = {"gene": entry.get("Gene", gene), "uniprot": uid}
                for t in HPA_TISSUES:
                    clean = t.replace("+", " ").title()
                    key = f"Tissue RNA - {t.replace('+', ' ')} [nTPM]"
                    row[clean] = float(entry.get(key, 0) or 0)
                records.append(row)
                print(f"    {gene}: OK")
        except Exception as e:
            print(f"    {gene}: FAILED ({e})")
            records.append({"gene": gene, "uniprot": uid})

    hpa_df = pd.DataFrame(records).fillna(0.0)
    tissue_cols = [c for c in hpa_df.columns if c not in ("gene", "uniprot")]

    # Map gene to short protein name
    gene_to_short = {}
    for col_name, short in PROTEIN_SHORT.items():
        uid = col_name.split("_")[0]
        g = UNIPROT_TO_GENE.get(uid, uid)
        gene_to_short[g] = short
    hpa_df["protein"] = hpa_df["gene"].map(gene_to_short).fillna(hpa_df["gene"])

    # Identify upregulated proteins
    merged = survey[["participant_id", "stress_cluster"]].merge(omics, on="participant_id")
    upreg = []
    for col in PROTEIN_COLS:
        m_low = merged.loc[merged["stress_cluster"] == 0, col].mean()
        m_high = merged.loc[merged["stress_cluster"] == merged["stress_cluster"].max(), col].mean()
        if m_high > m_low:
            uid = col.split("_")[0]
            if uid in UNIPROT_TO_GENE:
                upreg.append(UNIPROT_TO_GENE[uid])
    print(f"\n  Upregulated proteins: {len(upreg)}")

    # ── Heatmap: all proteins x tissues ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 8))
    heat_data = hpa_df.set_index("protein")[tissue_cols]
    heat_log = np.log10(heat_data + 1)
    sns.heatmap(heat_log, cmap="viridis", ax=ax, linewidths=0.3,
                cbar_kws={"label": "log₁₀(nTPM + 1)"})
    ax.set_title("RNA Expression Across Human Tissues (Human Protein Atlas)", fontsize=13)
    ax.set_ylabel("")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tissue_expression_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: tissue_expression_heatmap.png")

    # ── Bar: aggregate organ burden for upregulated proteins ─────────────
    upreg_df = hpa_df[hpa_df["gene"].isin(upreg)]
    if not upreg_df.empty:
        organ_sum = upreg_df[tissue_cols].sum().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.YlOrRd(np.linspace(0.2, 0.9, len(organ_sum)))
        ax.barh(organ_sum.index, organ_sum.values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Cumulative nTPM")
        ax.set_title("Cumulative Expression of Upregulated Proteins by Tissue")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "tissue_organ_burden.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: tissue_organ_burden.png")

    # Save data
    hpa_df.to_csv(OUTPUT_DIR / "hpa_tissue_expression.csv", index=False)
    print(f"  Saved: hpa_tissue_expression.csv")

    return hpa_df


# ════════════════════════════════════════════════════════════════════════════
# PROTEIN INTERACTION NETWORK (STRING DB)
# ════════════════════════════════════════════════════════════════════════════
def analyze_protein_network(survey, omics):
    """Fetch PPI from STRING DB and build an interaction network."""
    print("\n" + "=" * 72)
    print("PROTEIN INTERACTION NETWORK (STRING DB)")
    print("=" * 72)

    gene_names = list(UNIPROT_TO_GENE.values())
    identifiers = "%0d".join(gene_names)

    # Fetch internal network
    print("\n  Fetching interactions from string-db.org...")
    url = (f"https://string-db.org/api/json/network"
           f"?identifiers={identifiers}&species=9606&required_score=400")
    r = requests.get(url, timeout=20)
    internal = r.json() if r.status_code == 200 else []
    print(f"  Internal interactions: {len(internal)}")

    # Fetch partners
    url2 = (f"https://string-db.org/api/json/interaction_partners"
            f"?identifiers={identifiers}&species=9606&limit=5")
    r2 = requests.get(url2, timeout=20)
    partners = r2.json() if r2.status_code == 200 else []
    print(f"  Partner interactions: {len(partners)}")

    # Build graph
    G = nx.Graph()
    core_genes = set(gene_names)

    for gene in core_genes:
        for col, s in PROTEIN_SHORT.items():
            uid = col.split("_")[0]
            if UNIPROT_TO_GENE.get(uid) == gene:
                G.add_node(gene, display=s, node_type="core")
                break

    for edge in internal:
        a, b = edge["preferredName_A"], edge["preferredName_B"]
        if a in core_genes and b in core_genes:
            G.add_edge(a, b, score=edge["score"], edge_type="internal")

    for edge in partners:
        a, b = edge["preferredName_A"], edge["preferredName_B"]
        if a not in G:
            G.add_node(a, display=a, node_type="partner")
        if b not in G:
            G.add_node(b, display=b, node_type="partner")
        if not G.has_edge(a, b):
            G.add_edge(a, b, score=edge["score"], edge_type="partner")

    print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ── Network plot ─────────────────────────────────────────────────────
    pos = nx.spring_layout(G, k=1.8, iterations=80, seed=42)
    fig, ax = plt.subplots(figsize=(14, 12))

    # Draw edges
    internal_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "internal"]
    partner_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "partner"]

    nx.draw_networkx_edges(G, pos, edgelist=partner_edges, alpha=0.2,
                           edge_color="gray", width=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=internal_edges, alpha=0.5,
                           edge_color="#555555", width=1.5, ax=ax)

    # Draw nodes
    core_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "core"]
    partner_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "partner"]

    nx.draw_networkx_nodes(G, pos, nodelist=partner_nodes, node_size=200,
                           node_color="#88CCEE", alpha=0.7, edgecolors="gray", ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=core_nodes, node_size=600,
                           node_color="#CC6677", edgecolors="black", linewidths=1.5, ax=ax)

    # Labels
    labels = {n: G.nodes[n].get("display", n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight="bold", ax=ax)

    ax.set_title("Protein–Protein Interaction Network (STRING DB)\n"
                 "Red = core panel proteins, Blue = interaction partners", fontsize=13)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "protein_interaction_network.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: protein_interaction_network.png")

    # ── Centrality table ─────────────────────────────────────────────────
    degree_cent = nx.degree_centrality(G)
    between_cent = nx.betweenness_centrality(G)
    print("\n── Network Centrality (Core Proteins) ──")
    print(f"  {'Protein':15s} {'Degree':>8s} {'Deg.Cent.':>10s} {'Betw.Cent.':>10s}")
    for gene in sorted(core_genes):
        if gene in G:
            s = G.nodes[gene].get("display", gene)
            print(f"  {s:15s} {G.degree(gene):8d} {degree_cent[gene]:10.3f} {between_cent[gene]:10.3f}")

    # Save edge list
    edge_data = []
    for u, v, d in G.edges(data=True):
        edge_data.append({
            "protein_a": G.nodes[u].get("display", u),
            "protein_b": G.nodes[v].get("display", v),
            "score": d.get("score", 0),
            "type": d.get("edge_type", ""),
        })
    pd.DataFrame(edge_data).to_csv(OUTPUT_DIR / "protein_interactions.csv", index=False)
    print(f"\n  Saved: protein_interactions.csv")

    return G


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "▓" * 72)
    print("  SFRN ANALYSIS PIPELINE — Stress–CV Pathways")
    print("▓" * 72)

    # Step 1
    survey, geo, omics = step1_data_preparation()

    # Step 2
    survey = step2_clustering(survey)

    # Step 3
    merged_geo = step3_geospatial(survey, geo)

    # Step 4
    merged_omics = step4_proteomics(survey, omics)

    # CV disease prevalence
    plot_cv_prevalence(survey)

    # Geospatial heatmaps
    generate_geospatial_heatmaps(survey, geo)

    # Tissue expression (HPA)
    analyze_tissue_expression(survey, omics)

    # Protein interaction network (STRING)
    analyze_protein_network(survey, omics)

    # ── Save cluster assignments ─────────────────────────────────────────
    survey.to_csv(OUTPUT_DIR / "survey_with_clusters.csv", index=False)
    print(f"\n  Saved: survey_with_clusters.csv (cluster labels for downstream use)")

    print("\n" + "▓" * 72)
    print("  PIPELINE COMPLETE — All outputs in ./results/")
    print("▓" * 72 + "\n")


if __name__ == "__main__":
    main()
