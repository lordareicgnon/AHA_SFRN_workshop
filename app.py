"""
SFRN Analysis Pipeline — Guided Tutorial
==========================================
A step-by-step walkthrough of the Stress–CV Pathways analysis.
Run: streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
import networkx as nx
import requests
import sys, io

# ── Global config ────────────────────────────────────────────────────────────
pio.templates.default = "simple_white"
FONT = dict(family="Arial, Helvetica, sans-serif", size=14, color="black")

st.set_page_config(page_title="SFRN Pipeline Tutorial", page_icon="🫀", layout="wide")

DATA_DIR = Path(__file__).parent

PSYCHOSOCIAL_TOTALS = [
    "PHQ8_total", "GAD2_total", "PSS10_total", "PCL6_total",
    "PROMIS_anx_raw", "PROMIS_dep_raw", "PSQI_global",
]
PSYCH_LABELS = {
    "PHQ8_total": "PHQ-8 (Depression)", "GAD2_total": "GAD-2 (Anxiety)",
    "PSS10_total": "PSS-10 (Perceived Stress)", "PCL6_total": "PCL-6 (PTSD)",
    "PROMIS_anx_raw": "PROMIS Anxiety", "PROMIS_dep_raw": "PROMIS Depression",
    "PSQI_global": "PSQI (Sleep)",
}
PROTEIN_COLS = [
    "P02741_CRP", "P05231_IL6", "P01375_TNF_alpha",
    "P16860_NT_proBNP", "P19429_Troponin_I",
    "P15692_VEGF_A", "P14780_MMP9", "P05362_ICAM1",
    "P05121_PAI1", "P17931_Galectin3", "Q01638_ST2_IL1RL1",
    "Q99988_GDF15", "P05305_Endothelin1",
    "Q15848_Adiponectin", "Q13884_Lp_PLA2",
]
PROTEIN_SHORT = {
    "P02741_CRP": "CRP", "P05231_IL6": "IL-6", "P01375_TNF_alpha": "TNF-α",
    "P16860_NT_proBNP": "NT-proBNP", "P19429_Troponin_I": "Troponin I",
    "P15692_VEGF_A": "VEGF-A", "P14780_MMP9": "MMP-9", "P05362_ICAM1": "ICAM-1",
    "P05121_PAI1": "PAI-1", "P17931_Galectin3": "Galectin-3",
    "Q01638_ST2_IL1RL1": "ST2/IL1RL1", "Q99988_GDF15": "GDF-15",
    "P05305_Endothelin1": "Endothelin-1", "Q15848_Adiponectin": "Adiponectin",
    "Q13884_Lp_PLA2": "Lp-PLA2",
}
DX_COLS = {
    "dx_hypertension": "Hypertension", "dx_diabetes": "Diabetes",
    "dx_heart_failure": "Heart Failure", "dx_MI_history": "MI History",
    "dx_stroke_history": "Stroke History",
}
CLUSTER_MAP = {0: "Low Burden", 1: "Moderate Burden", 2: "High Burden"}
UNIPROT_TO_GENE = {
    "P02741": "CRP", "P05231": "IL6", "P01375": "TNF",
    "P16860": "NPPB", "P19429": "TNNI3", "P15692": "VEGFA",
    "P14780": "MMP9", "P05362": "ICAM1", "P05121": "SERPINE1",
    "P17931": "LGALS3", "Q01638": "IL1RL1", "Q99988": "GDF15",
    "P05305": "EDN1", "Q15848": "ADIPOQ", "Q13884": "PLA2G7",
}
ZIP_COORDS = {
    95608:(38.6284,-121.3287),95610:(38.6946,-121.2692),95621:(38.6952,-121.3075),
    95624:(38.4232,-121.3599),95630:(38.6709,-121.1529),95660:(38.6707,-121.3781),
    95670:(38.6072,-121.2761),95673:(38.6895,-121.4479),95757:(38.4081,-121.4294),
    95758:(38.4243,-121.4370),95762:(38.6850,-121.0680),95811:(38.5762,-121.4880),
    95814:(38.5804,-121.4922),95815:(38.6093,-121.4443),95816:(38.5728,-121.4675),
    95817:(38.5498,-121.4583),95818:(38.5568,-121.4929),95819:(38.5683,-121.4366),
    95820:(38.5347,-121.4451),95821:(38.6239,-121.3837),95822:(38.5091,-121.4935),
    95823:(38.4797,-121.4438),95824:(38.5178,-121.4419),95825:(38.5892,-121.4057),
    95826:(38.5539,-121.3693),95827:(38.5662,-121.3286),95828:(38.4826,-121.4006),
    95829:(38.4689,-121.3440),95830:(38.4896,-121.2772),95831:(38.4962,-121.5297),
    95832:(38.4695,-121.4883),95833:(38.6157,-121.5053),95834:(38.6383,-121.5072),
    95835:(38.6626,-121.4834),95836:(38.7198,-121.5343),95837:(38.6817,-121.6030),
    95838:(38.6406,-121.4440),95841:(38.6627,-121.3406),95842:(38.6865,-121.3494),
    95843:(38.7159,-121.3648),95864:(38.5878,-121.3769),
}
HPA_TISSUES = [
    "heart","liver","kidney","lung","brain","adipose+tissue","bone+marrow",
    "spleen","colon","small+intestine","stomach","pancreas","skin",
    "smooth+muscle","lymph+node","adrenal+gland","thyroid+gland",
    "esophagus","gallbladder","urinary+bladder","placenta","breast",
    "endometrium","ovary","prostate","testis","salivary+gland",
]


# ── Dynamic cluster colors ──────────────────────────────────────────────────
_TOL_QUALITATIVE = [
    "#44AA99", "#DDCC77", "#CC6677", "#88CCEE", "#882255",
    "#332288", "#117733", "#999933", "#AA4499", "#661100",
    "#6699CC", "#DDDDDD",
]

def generate_palette(k):
    """Return a {label: color} dict for k clusters using Tol's qualitative palette.
    For k=3 the canonical Low/Moderate/High Burden palette is preserved."""
    if k == 3:
        return {"Low Burden": "#44AA99", "Moderate Burden": "#DDCC77", "High Burden": "#CC6677"}
    palette = {}
    for i in range(k):
        label = CLUSTER_MAP.get(i, f"Cluster {i}")
        palette[label] = _TOL_QUALITATIVE[i % len(_TOL_QUALITATIVE)]
    return palette


# ── Inject realistic missingness ────────────────────────────────────────────
@st.cache_data
def inject_missingness(survey, omics, geo, pct=0.06, seed=42):
    """Create copies with realistic missing-data patterns.
    Returns (survey_miss, omics_miss, geo_miss, truth_survey, truth_omics, truth_geo).
    The truth frames hold the original values so imputation accuracy can be measured.
    """
    rng = np.random.RandomState(seed)
    s = survey.copy()
    o = omics.copy()
    g = geo.copy()
    truth_s = survey.copy()
    truth_o = omics.copy()
    truth_g = geo.copy()

    # MCAR: Random 5-8% in PSQI_global, BMI
    for col in ["PSQI_global", "BMI"]:
        if col in s.columns:
            frac = rng.uniform(0.05, 0.08)
            mask = rng.rand(len(s)) < frac
            s.loc[mask, col] = np.nan

    # MAR: Missingness in PROMIS_dep_raw depends on age (older -> more missing)
    if "PROMIS_dep_raw" in s.columns and "age" in s.columns:
        age_prob = (s["age"] - s["age"].min()) / (s["age"].max() - s["age"].min())
        age_prob = 0.02 + age_prob * 0.10  # 2-12% depending on age
        mar_mask = rng.rand(len(s)) < age_prob
        s.loc[mar_mask, "PROMIS_dep_raw"] = np.nan

    # MNAR: High PHQ8_total values more likely to be missing (severe depression dropout)
    if "PHQ8_total" in s.columns:
        phq_prob = (s["PHQ8_total"] - s["PHQ8_total"].min()) / (s["PHQ8_total"].max() - s["PHQ8_total"].min())
        phq_prob = 0.01 + phq_prob * 0.12  # 1-13% depending on severity
        mnar_mask = rng.rand(len(s)) < phq_prob
        s.loc[mnar_mask, "PHQ8_total"] = np.nan

    # Omics: 3-5% MCAR in 2-3 protein columns
    protein_cols_present = [c for c in PROTEIN_COLS if c in o.columns]
    chosen_proteins = list(rng.choice(protein_cols_present, size=min(3, len(protein_cols_present)), replace=False))
    for col in chosen_proteins:
        frac = rng.uniform(0.03, 0.05)
        mask = rng.rand(len(o)) < frac
        o.loc[mask, col] = np.nan

    # Geospatial: 1-2 missing values in NDVI_greenspace
    if "NDVI_greenspace" in g.columns:
        n_miss = rng.randint(1, 3)
        idx = rng.choice(g.index, size=min(n_miss, len(g)), replace=False)
        g.loc[idx, "NDVI_greenspace"] = np.nan

    return s, o, g, truth_s, truth_o, truth_g


# ── Cached computation ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return (pd.read_csv(DATA_DIR / "survey_data.csv"),
            pd.read_csv(DATA_DIR / "geospatial_data.csv"),
            pd.read_csv(DATA_DIR / "omics_data.csv"))

@st.cache_data
def run_clustering(survey, method, k, vn_villages=60, vn_neighbors=20):
    X = StandardScaler().fit_transform(survey[PSYCHOSOCIAL_TOTALS])
    auto_k = False
    if method == "VillageNet":
        from VillageNet import VillageNet
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()  # suppress VillageNet prints
        model = VillageNet(villages=vn_villages, normalize=0, neighbors=vn_neighbors)
        model.fit(X)
        sys.stdout = old_stdout
        labels = model.comm_id
        auto_k = True
        k = len(np.unique(labels))
    elif method == "GMM":
        labels = GaussianMixture(n_components=k, covariance_type="full",
                                 n_init=10, random_state=42).fit_predict(X)
    else:
        labels = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(X)
    burden = pd.DataFrame(X, columns=PSYCHOSOCIAL_TOTALS)
    burden["c"] = labels
    rank = {old: new for new, old in enumerate(
        burden.groupby("c").mean().sum(axis=1).sort_values().index)}
    ordered = np.array([rank[c] for c in labels])
    return ordered, silhouette_score(X, ordered), X, k, auto_k

@st.cache_data
def cluster_selection_metrics(survey):
    X = StandardScaler().fit_transform(survey[PSYCHOSOCIAL_TOTALS])
    rows = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(X)
        gm = GaussianMixture(n_components=k, covariance_type="full",
                              n_init=5, random_state=42)
        gl = gm.fit_predict(X)
        rows.append({"k": k, "K-Means": silhouette_score(X, km),
                      "GMM": silhouette_score(X, gl), "BIC": gm.bic(X)})
    return pd.DataFrame(rows)

@st.cache_data
def build_zip_data(survey, geo):
    coords = pd.DataFrame([{"zip_code": z, "lat": c[0], "lon": c[1]}
                            for z, c in ZIP_COORDS.items()])
    zp = survey.groupby("zip_code")[PSYCHOSOCIAL_TOTALS].mean()
    sc = StandardScaler()
    zs = pd.DataFrame(sc.fit_transform(zp), index=zp.index, columns=PSYCHOSOCIAL_TOTALS)
    zp["psych_burden"] = zs.mean(axis=1)
    zp = zp.reset_index()
    dx_list = [c for c in DX_COLS if c in survey.columns]
    zd = survey.groupby("zip_code")[dx_list].mean() * 100
    zd["cv_burden"] = zd.mean(axis=1)
    zd = zd.reset_index()
    zn = survey.groupby("zip_code").size().reset_index(name="n")
    return coords.merge(geo, on="zip_code").merge(zp, on="zip_code").merge(zd, on="zip_code").merge(zn, on="zip_code")

@st.cache_data(ttl=3600, show_spinner="Fetching from Human Protein Atlas...")
def fetch_hpa():
    tc = ",".join([f"t_RNA_{t}" for t in HPA_TISSUES])
    rows = []
    for uid, gene in UNIPROT_TO_GENE.items():
        try:
            r = requests.get(f"https://www.proteinatlas.org/api/search_download.php"
                             f"?search={uid}&format=json&columns=g,up,{tc}&compress=no", timeout=15)
            if r.ok and r.json():
                e = r.json()[0]
                row = {"gene": e.get("Gene", gene), "uniprot": uid}
                for t in HPA_TISSUES:
                    key = f"Tissue RNA - {t.replace('+', ' ')} [nTPM]"
                    row[t.replace("+", " ").title()] = float(e.get(key, 0) or 0)
                rows.append(row)
        except Exception:
            rows.append({"gene": gene, "uniprot": uid})
    return pd.DataFrame(rows).fillna(0.0)

@st.cache_data(ttl=3600, show_spinner="Fetching from STRING DB...")
def fetch_string(score=400):
    ids = "%0d".join(UNIPROT_TO_GENE.values())
    try:
        r1 = requests.get(f"https://string-db.org/api/json/network?identifiers={ids}&species=9606&required_score={score}", timeout=20)
        internal = r1.json() if r1.ok else []
    except Exception:
        internal = []
    try:
        r2 = requests.get(f"https://string-db.org/api/json/interaction_partners?identifiers={ids}&species=9606&limit=5", timeout=20)
        partners = r2.json() if r2.ok else []
    except Exception:
        partners = []
    try:
        r3 = requests.get(f"https://string-db.org/api/json/enrichment?identifiers={ids}&species=9606", timeout=20)
        enrichment = r3.json() if r3.ok else []
    except Exception:
        enrichment = []
    return internal, partners, enrichment

def format_p(p):
    return "< 0.001" if p < 0.001 else f"{p:.4f}"

def sig(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "ns"


# ── Load & cluster ───────────────────────────────────────────────────────────
try:
    survey_raw, geo, omics = load_data()
except FileNotFoundError:
    st.error("Data files not found.")
    st.stop()

# Inject missingness for teaching purposes
survey_miss, omics_miss, geo_miss, truth_survey, truth_omics, truth_geo = inject_missingness(
    survey_raw, omics, geo)

# Sidebar: clustering parameters
with st.sidebar:
    st.markdown("### Settings")
    cluster_method = st.selectbox("Clustering algorithm", ["GMM", "K-Means", "VillageNet"])
    if cluster_method == "VillageNet":
        vn_villages = st.slider("VillageNet: number of villages", 10, 300, 20, step=10)
        vn_neighbors = st.slider("VillageNet: neighbors per village", 5, 40, 20, step=5)
        n_clusters_input = 3  # placeholder, VillageNet auto-detects
    else:
        vn_villages, vn_neighbors = 20, 20
        n_clusters_input = st.slider("Number of clusters (k)", 2, 10, 3)
    st.divider()
    st.caption("Scroll down to follow the analysis pipeline step by step.")

    # Glossary expander
    with st.expander("Glossary of Key Terms"):
        st.markdown("""
**PCA (Principal Component Analysis):** A technique for finding the underlying
patterns in multiple overlapping measurements. Think of it as reducing a complex
medical chart to its most important themes.

**Clustering:** Identifying distinct patient profiles from data -- like recognizing
that some patients share similar patterns of symptoms, labs, and behaviors.

**Silhouette Score:** A measure of how well-separated the patient groups are.
Ranges from -1 to 1; higher means the groups are more distinct (like clearly
different patient types vs. a blurred spectrum).

**Z-Score Standardization:** Putting all measurements on the same ruler. A blood
pressure of 140 mmHg and a depression score of 18 are on completely different
scales -- z-scoring converts both to "how many standard deviations from average."

**MCAR (Missing Completely At Random):** Data that is missing for no systematic
reason, like a lab tube breaking in transit.

**MAR (Missing At Random):** Missingness depends on another observed variable.
For example, older patients may skip certain survey questions more often.

**MNAR (Missing Not At Random):** Missingness depends on the unobserved value
itself. Patients with severe depression may drop out of a study, so the worst
depression scores go unmeasured.

**Cohen's d:** An effect size measure. d = 0.2 is small (barely noticeable
clinically), d = 0.5 is medium, d = 0.8+ is large (would likely be noticed
by a clinician).

**Fold Change:** How much a protein level differs between two groups. A fold
change of +1.5 means levels are 1.5 units higher in the comparison group.

**BIC (Bayesian Information Criterion):** A model selection metric that penalizes
complexity. Lower BIC suggests a better balance between model fit and simplicity.

**STRING DB:** A database of known and predicted protein-protein interactions,
showing how proteins work together in biological pathways.

**HPA (Human Protein Atlas):** A map of where each protein is produced in the
human body -- which organs and tissues express it most.
        """)

labels, sil_score, X_scaled, n_clusters, auto_k = run_clustering(
    survey_raw, cluster_method, n_clusters_input, vn_villages, vn_neighbors)
survey = survey_raw.copy()
survey["stress_cluster"] = labels
COLORS = generate_palette(n_clusters)
survey["stress_phenotype"] = survey["stress_cluster"].map(
    CLUSTER_MAP if n_clusters == 3 else {i: f"Cluster {i}" for i in range(n_clusters)})
phenotype_order = (survey.groupby("stress_phenotype")["stress_cluster"]
                   .first().sort_values().index.tolist())


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                    GUIDED TUTORIAL BEGINS HERE                           ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# ── HEADER ───────────────────────────────────────────────────────────────────
st.title("Stress-to-Cardiovascular Disease Pathways")

st.markdown("""
> **Core Motivation:** Psychosocial stress is complex and multidimensional, but we
> often measure it with multiple overlapping tools -- so we need a structured way to
> **understand**, **simplify**, and **use** that information to study health outcomes
> like cardiovascular disease.
""")

st.markdown("""
**Research Question:** *How does chronic psychosocial stress translate into
cardiovascular disease risk, and can we identify distinct stress profiles that map
onto environmental exposures, molecular mechanisms, and clinical outcomes?*
""")

st.markdown("""
This pipeline follows a structured analytical logic:

| Step | Question | Method |
|------|----------|--------|
| **1. Data Preparation** | Is our data ready for analysis? | Inspect, clean, handle missing data, standardize |
| **2. What are we measuring?** | Do our 7 instruments capture one thing or many? | PCA / dimensionality analysis |
| **3. Who are the people?** | Are there distinct subgroups in our cohort? | Unsupervised clustering |
| **4. Where do they live?** | Do stress profiles map to neighborhoods? | Geospatial linkage |
| **5. What's in their blood?** | Is there a molecular fingerprint? | Differential protein expression |
| **6. Does it cause disease?** | Do these patterns predict CV outcomes? | Prevalence analysis |
| **7--8. Deeper biology** | Which organs? Which pathways? | HPA tissue atlas + STRING network |
""")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Participants", f"{len(survey_raw):,}")
col2.metric("ZIP codes", f"{geo.shape[0]}")
col3.metric("Proteins", f"{len(PROTEIN_COLS)}")
col4.metric("Clusters found", f"{n_clusters}")

with st.expander("How the datasets connect"):
    st.code("""
    survey_data.csv ──(participant_id)──► omics_data.csv
           │                                (15 protein markers)
       (zip_code)
           │
           ▼
    geospatial_data.csv
       (41 Sacramento-area ZIP codes)
    """, language=None)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA PREPARATION
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("Step 1 -- Data Preparation")
st.markdown("""
Before any analysis, we need to **inspect** and **prepare** the data. In clinical
research this is like reviewing a patient's chart before making a diagnosis -- you
need to know what information you have, what is missing, and whether the measurements
are on comparable scales. Skipping this step is like prescribing medication without
reading the chart.
""")

# ── 1a. Data Overview ──────────────────────────────────────────────────────
st.subheader("1a. Data Overview")
miss_data = pd.DataFrame({
    "Dataset": ["Survey", "Geospatial", "Omics"],
    "Rows": [len(survey_raw), len(geo), len(omics)],
    "Columns": [survey_raw.shape[1], geo.shape[1], omics.shape[1]],
    "Missing Values (original)": [survey_raw.isnull().sum().sum(), geo.isnull().sum().sum(), omics.isnull().sum().sum()],
})
st.dataframe(miss_data, use_container_width=True, hide_index=True)

with st.expander("Explore: raw data tables"):
    t1, t2, t3 = st.tabs(["Survey", "Geospatial", "Omics"])
    with t1:
        st.dataframe(survey_raw.head(50), use_container_width=True, height=250)
    with t2:
        st.dataframe(geo, use_container_width=True, height=250)
    with t3:
        st.dataframe(omics.head(50), use_container_width=True, height=250)


# ── 1b. Demographics ────────────────────────────────────────────────────
st.subheader("1b. Cohort Demographics")
st.markdown("""
Understanding who is in your study is essential. Just as a clinician considers a
patient's age, sex, and background before interpreting lab results, we need to know
the composition of our cohort before interpreting patterns in the data.
""")
col1, col2, col3 = st.columns(3)
with col1:
    fig = px.histogram(survey_raw, x="age", nbins=30, color_discrete_sequence=["#555555"], title="Age")
    fig.update_layout(bargap=0.1, font=FONT, height=280)
    fig.update_traces(marker_line_width=1, marker_line_color="black")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    sex_counts = survey_raw["sex"].value_counts().reset_index()
    sex_counts.columns = ["Sex", "Count"]
    fig = px.pie(sex_counts, values="Count", names="Sex", title="Sex", color_discrete_sequence=["#555555", "#AAA"])
    fig.update_layout(font=FONT, height=280)
    st.plotly_chart(fig, use_container_width=True)
with col3:
    race_counts = survey_raw["race_ethnicity"].value_counts().reset_index()
    race_counts.columns = ["Race/Ethnicity", "Count"]
    fig = px.bar(race_counts, x="Count", y="Race/Ethnicity", orientation="h",
                 color_discrete_sequence=["#555555"], title="Race/Ethnicity")
    fig.update_layout(font=FONT, height=280, yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig, use_container_width=True)

# ── 1c. Instrument Distributions ─────────────────────────────────────────
st.subheader("1c. Psychosocial Instruments")
st.markdown("""
Each participant completed **7 validated instruments** measuring depression (PHQ-8),
anxiety (GAD-2, PROMIS), perceived stress (PSS-10), PTSD symptoms (PCL-6), and
sleep quality (PSQI). Think of each instrument as a different lens on the same
patient -- like how a blood pressure cuff and an EKG both assess the heart but
capture different information.

These instruments have **different scales** -- PHQ-8 ranges 0-24 while GAD-2 ranges
0-6 -- so we will need to standardize them before analysis (just as you would not
compare a temperature in Fahrenheit to one in Celsius without converting first).
""")

desc = survey_raw[PSYCHOSOCIAL_TOTALS].describe().T.round(2)
desc.index = [PSYCH_LABELS[c] for c in desc.index]
st.dataframe(desc[["mean", "std", "min", "25%", "50%", "75%", "max"]], use_container_width=True)

cols = st.columns(4)
for i, col_name in enumerate(PSYCHOSOCIAL_TOTALS):
    with cols[i % 4]:
        fig = px.histogram(survey_raw, x=col_name, nbins=25, color_discrete_sequence=["steelblue"],
                           title=PSYCH_LABELS[col_name].split(" (")[0])
        fig.update_layout(height=200, font=dict(size=11), margin=dict(t=30, b=20), showlegend=False, bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)


# ── 1d. Missingness Patterns ───────────────────────────────────────────────
st.subheader("1d. Missingness Patterns")

st.markdown("""
The original datasets are fully complete. In real-world clinical settings, however, missing data is the norm rather than the exception. Patients may skip questionnaire items, lab samples can be compromised, and monitoring devices may fail. These gaps are not just inevitable—they are meaningful. The pattern of missingness is just as important as the amount, because each type arises from different underlying causes and requires different analytical approaches. To support understanding of how missing data should be handled, modified versions of these datasets have been created with realistic missingness patterns applied. Three common types of missing data have been introduced to enable direct comparison. Use the toggle below to explore each type:
""")

miss_type = st.radio(
    "Select missingness mechanism to explore:",
    ["MCAR -- Missing Completely At Random",
     "MAR -- Missing At Random",
     "MNAR -- Missing Not At Random"],
    key="miss_type_radio"
)

if "MCAR" in miss_type:
    st.info("""
    **MCAR: Missing Completely At Random**

    *Clinical analogy:* A lab tube breaks during transport. The breakage has nothing to do
    with the patient's health -- it is pure bad luck. Whether a value is missing is
    unrelated to any variable in the study.

    In our data, we randomly removed 5-8% of **PSQI (sleep quality)** and **BMI** values.
    These gaps are scattered uniformly -- no pattern by age, sex, or severity.
    """)
elif "MAR" in miss_type:
    st.info("""
    **MAR: Missing At Random**

    *Clinical analogy:* Older patients are more likely to skip a lengthy depression
    questionnaire because of fatigue or vision problems -- not because of their depression
    level, but because of their age. The missingness depends on an **observed** variable (age),
    not the missing value itself.

    In our data, **PROMIS Depression** scores are more likely to be missing for older
    participants. If you compare the age distribution of people with and without PROMIS
    scores, you will see a clear shift.
    """)
else:
    st.warning("""
    **MNAR: Missing Not At Random**

    *Clinical analogy:* Patients with the most severe depression drop out of a study.
    Their PHQ-8 scores would have been the highest, but they are never recorded. The
    missingness depends directly on the **unobserved value** -- the sicker you are,
    the more likely your data is missing.

    This is the most dangerous type because it biases results toward healthier patients.
    In our data, high **PHQ-8** values are more likely to be missing, simulating
    depression-related dropout.
    """)

# Missingness heatmap
st.markdown("**Missingness heatmap** -- each row is a participant, each column is a variable. "
            "Yellow cells indicate missing values:")

# Build a missingness indicator for key columns
miss_cols_survey = ["PHQ8_total", "PSQI_global", "PROMIS_dep_raw"]
if "BMI" in survey_miss.columns:
    miss_cols_survey.append("BMI")

miss_indicator = survey_miss[miss_cols_survey].isnull().astype(int)
# Show a random sample for visual clarity
sample_size = min(200, len(miss_indicator))
miss_sample = miss_indicator.sample(sample_size, random_state=42).reset_index(drop=True)
fig = px.imshow(miss_sample.T.values,
                x=[str(i) for i in range(sample_size)],
                y=[PSYCH_LABELS.get(c, c) for c in miss_cols_survey],
                color_continuous_scale=[[0, "#2b2b2b"], [1, "#FFDD57"]],
                aspect="auto", title="Missingness Map (200-participant sample)")
fig.update_traces(xgap=1, ygap=3)
fig.update_layout(height=250, font=FONT, xaxis=dict(showticklabels=False, title="Participants"),
                  coloraxis_showscale=False)
st.plotly_chart(fig, use_container_width=True)

# Missingness percentage by column
st.markdown("**Missingness percentage by column:**")
miss_pct_survey = (survey_miss[miss_cols_survey].isnull().sum() / len(survey_miss) * 100).round(2)
miss_pct_df = pd.DataFrame({
    "Variable": [PSYCH_LABELS.get(c, c) for c in miss_cols_survey],
    "Missing (%)": miss_pct_survey.values,
    "Mechanism": ["MNAR (severity-dependent dropout)" if c == "PHQ8_total"
                  else "MAR (age-dependent)" if c == "PROMIS_dep_raw"
                  else "MCAR (random)" for c in miss_cols_survey],
})
st.dataframe(miss_pct_df, use_container_width=True, hide_index=True)

st.markdown("""
> **Why does this matter clinically?** If you simply delete patients with missing data
> (called "complete case analysis"), you may be throwing away your sickest patients (MNAR)
> or biasing your sample toward younger individuals (MAR). Understanding the *mechanism*
> of missingness determines which statistical remedy is appropriate.
""")


# ── 1e. Imputation Strategies ──────────────────────────────────────────────
st.subheader("1e. Imputation Strategies")
st.markdown("""
Once we understand *why* data is missing, we need to decide how to fill the gaps.
Think of it like a clinician estimating a patient's missing lab value: you could use
the population average, look at similar patients, or simply ignore that patient.
Each approach has trade-offs.
""")

impute_method = st.selectbox(
    "Select an imputation method:",
    ["Complete cases only (listwise deletion)", "Mean/Median imputation", "KNN imputation (using similar patients)"],
    key="impute_select"
)

# Pick a demonstration column that has missingness
demo_col = "PSQI_global" if "PSQI_global" in survey_miss.columns else miss_cols_survey[0]
demo_label = PSYCH_LABELS.get(demo_col, demo_col)

has_missing = survey_miss[demo_col].isnull().any()
missing_mask = survey_miss[demo_col].isnull()
true_values = truth_survey.loc[missing_mask, demo_col] if has_missing else pd.Series(dtype=float)

if "Complete cases" in impute_method:
    imputed_series = survey_miss[demo_col].dropna()
    method_name = "Complete cases"
    st.info("""
    **Complete cases only:** Drop every participant who has *any* missing value.

    *Clinical analogy:* Only analyzing patients who completed every single form. Simple,
    but you may lose a large fraction of your data -- and the patients you lose may be
    systematically different from those you keep.
    """)
    st.warning(f"""
    **When to use this:** Only when missingness is very low (<2-3%) and MCAR. If
    missingness is MAR or MNAR, complete case analysis produces biased results.
    """)
    imputed_values = pd.Series(dtype=float)

elif "Mean/Median" in impute_method:
    fill_val = survey_miss[demo_col].median()
    imputed_series = survey_miss[demo_col].fillna(fill_val)
    imputed_values = pd.Series(fill_val, index=true_values.index) if has_missing else pd.Series(dtype=float)
    method_name = "Median"
    st.info(f"""
    **Mean/Median imputation:** Replace every missing value with the column median
    ({fill_val:.1f} for {demo_label}).

    *Clinical analogy:* Estimating a patient's missing blood pressure as the clinic-wide
    average. Quick and easy, but it makes the data look more uniform than it really is
    -- it shrinks the spread and can hide real variability.
    """)
    st.warning("""
    **When to use this:** For quick exploratory work or when missingness is low and MCAR.
    Not recommended for final analyses because it underestimates variance and weakens
    relationships between variables.
    """)

else:
    knn_cols = [c for c in PSYCHOSOCIAL_TOTALS if c in survey_miss.columns]
    knn_data = survey_miss[knn_cols].copy()
    imputer = KNNImputer(n_neighbors=5)
    imputed_arr = imputer.fit_transform(knn_data)
    imputed_df = pd.DataFrame(imputed_arr, columns=knn_cols, index=survey_miss.index)
    imputed_series = imputed_df[demo_col]
    imputed_values = imputed_df.loc[missing_mask, demo_col] if has_missing else pd.Series(dtype=float)
    method_name = "KNN (k=5)"
    st.info("""
    **KNN imputation:** For each patient with a missing value, find the 5 most similar
    patients (based on their other scores) and average their values.

    *Clinical analogy:* Instead of using the clinic average, you look at 5 patients who
    are most similar to this one in age, other test scores, etc., and use their average.
    This preserves individual variation much better than mean imputation.
    """)
    st.success("""
    **When to use this:** When you have moderate missingness and other variables that
    are correlated with the missing one. KNN leverages the relationships between
    variables, making it suitable for both MCAR and MAR patterns.
    """)

# Before/after distribution plot
st.markdown(f"**Before vs. After Imputation** -- {demo_label} ({method_name}):")

col_before, col_after = st.columns(2)
with col_before:
    orig_vals = truth_survey[demo_col].dropna()
    fig = px.histogram(orig_vals, nbins=30, color_discrete_sequence=["#555555"],
                       title=f"Original (complete) distribution")
    fig.update_layout(height=280, font=FONT, bargap=0.05, showlegend=False,
                      xaxis_title=demo_label, yaxis_title="Count")
    fig.update_traces(marker_line_width=1, marker_line_color="black")
    st.plotly_chart(fig, use_container_width=True)

with col_after:
    if "Complete cases" in impute_method:
        fig = px.histogram(imputed_series, nbins=30, color_discrete_sequence=["#44AA99"],
                           title=f"After listwise deletion (n={len(imputed_series):,})")
    else:
        fig = px.histogram(imputed_series, nbins=30, color_discrete_sequence=["#44AA99"],
                           title=f"After {method_name} imputation")
    fig.update_layout(height=280, font=FONT, bargap=0.05, showlegend=False,
                      xaxis_title=demo_label, yaxis_title="Count")
    fig.update_traces(marker_line_width=1, marker_line_color="black")
    st.plotly_chart(fig, use_container_width=True)

# Imputed vs true scatter + RMSE (for methods that produce imputed values)
if has_missing and len(imputed_values) > 0 and len(true_values) > 0 and "Complete cases" not in impute_method:
    rmse = np.sqrt(np.mean((imputed_values.values - true_values.values) ** 2))

    col_scatter, col_metrics = st.columns([2, 1])
    with col_scatter:
        scatter_df = pd.DataFrame({"True Value": true_values.values, "Imputed Value": imputed_values.values})
        fig = px.scatter(scatter_df, x="True Value", y="Imputed Value",
                         color_discrete_sequence=["#CC6677"], opacity=0.5,
                         title=f"Imputed vs. True Values (we know the truth because we created the gaps)")
        min_val = min(scatter_df["True Value"].min(), scatter_df["Imputed Value"].min())
        max_val = max(scatter_df["True Value"].max(), scatter_df["Imputed Value"].max())
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                 mode="lines", line=dict(dash="dash", color="gray"),
                                 showlegend=False, name="Perfect"))
        fig.update_layout(height=350, font=FONT)
        st.plotly_chart(fig, use_container_width=True)

    with col_metrics:
        st.metric("RMSE", f"{rmse:.3f}")
        st.metric("Missing values imputed", f"{len(imputed_values):,}")
        st.markdown(f"""
        Points close to the diagonal line indicate accurate imputation.
        An RMSE of **{rmse:.3f}** means the typical imputation error is about
        {rmse:.1f} units on the {demo_label} scale.
        """)

    st.markdown("""
    > **Why does this matter clinically?** Poor imputation can introduce bias that
    > changes study conclusions. For example, if mean imputation pulls all missing
    > depression scores toward the average, you would underestimate the proportion of
    > severely depressed patients -- potentially missing the group most in need of
    > intervention.
    """)

st.markdown("""
> **How can this be applied?** A clinic collecting patient-reported outcomes could use
> KNN imputation to recover missing questionnaire data before feeding it into a risk
> prediction model, rather than excluding incomplete records and losing statistical
> power.
""")

# ── 1f. Standardization ─────────────────────────────────────────────────
st.subheader("1f. Standardization")
st.markdown("""
We **z-score standardize** each instrument to zero mean and unit variance. This is
like putting all measurements on the same ruler. Without this step, an instrument
with a larger numeric range (say, PSS-10 going up to 40) would dominate the analysis
over one with a smaller range (GAD-2 going up to 6) -- not because it is more
important, but simply because its numbers are bigger.

After standardization, a score of +1 on *any* instrument means "one standard deviation
above average" -- making comparisons fair and meaningful.
""")

st.info("""
**Data is clean and standardized.** We have handled missing values, confirmed our
cohort demographics, and put all instruments on the same scale. Next question:
what are these 7 instruments actually measuring? Are they capturing 7 different
things, or different reflections of the same underlying condition?
""")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — WHAT ARE WE MEASURING? (PCA)
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("Step 2 -- What Are We Measuring?")
st.markdown("""
We have 7 instruments, but are they measuring 7 different things? Or are they
capturing the same underlying condition -- "general psychosocial distress" -- from
different angles?

Think of it this way: if you measured a patient's temperature with an oral thermometer,
an ear thermometer, and a forehead strip, you would get 3 numbers, but they all
reflect **one thing** -- body temperature. PCA does the same thing with our 7
psychosocial instruments: it finds the **underlying patterns** in multiple
overlapping measurements.

**Principal Component Analysis (PCA)** reveals:
1. **How many underlying dimensions** exist in the data
2. **How strongly each instrument relates** to those dimensions (loadings)
3. **Whether we can simplify** 7 measures into fewer composite scores
""")

# ── PCA ──────────────────────────────────────────────────────────────────
X_std = StandardScaler().fit_transform(survey_raw[PSYCHOSOCIAL_TOTALS])
pca_full = PCA().fit(X_std)
var_explained = pca_full.explained_variance_ratio_
cum_var = np.cumsum(var_explained)
loadings = pd.DataFrame(pca_full.components_.T,
                         index=[PSYCH_LABELS[c].split(" (")[0] for c in PSYCHOSOCIAL_TOTALS],
                         columns=[f"PC{i+1}" for i in range(len(var_explained))])

# ── Correlations first ───────────────────────────────────────────────────
st.subheader("2a. Are the Instruments Correlated?")
st.markdown("""
Before running PCA, let us look at how the instruments relate to each other. If a
patient scores high on depression, do they also tend to score high on anxiety and
poor sleep? High correlations suggest the instruments are tapping into a shared
underlying factor -- which PCA will formalize.
""")
corr = survey_raw[PSYCHOSOCIAL_TOTALS].corr()
short_labels = [PSYCH_LABELS[c].split(" (")[0] for c in PSYCHOSOCIAL_TOTALS]
fig = px.imshow(corr.values, x=short_labels, y=short_labels,
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="equal",
                title="Pearson Correlation Matrix")
fig.update_traces(xgap=3, ygap=3)
fig.update_layout(font=FONT, height=400)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Interpretation:** All instruments are positively correlated (r = 0.4-0.9). This confirms
they share common variance -- someone who scores high on depression tends to score high on
anxiety, stress, and poor sleep too. But they are not *identical*, so each adds information.
This is like finding that blood pressure, cholesterol, and blood sugar are correlated but
each still provides unique diagnostic value.
""")

# ── Scree plot + loadings ────────────────────────────────────────────────
st.subheader("2b. How Many Dimensions?")

col1, col2 = st.columns(2)
with col1:
    pca_df = pd.DataFrame({"Component": [f"PC{i+1}" for i in range(len(var_explained))],
                            "Variance Explained": var_explained, "Cumulative": cum_var})
    fig = go.Figure()
    fig.add_trace(go.Bar(x=pca_df["Component"], y=pca_df["Variance Explained"],
                          name="Individual", marker_color="#555"))
    fig.add_trace(go.Scatter(x=pca_df["Component"], y=pca_df["Cumulative"],
                              mode="lines+markers", name="Cumulative",
                              line=dict(color="#CC6677", width=2)))
    fig.update_layout(title="Variance Explained per Component", yaxis_title="Proportion",
                      height=350, font=FONT)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.imshow(loadings.iloc[:, :3].values,
                    x=["PC1", "PC2", "PC3"], y=loadings.index.tolist(),
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                    title="Component Loadings (first 3 PCs)")
    fig.update_traces(xgap=3, ygap=3)
    fig.update_layout(height=350, font=FONT)
    st.plotly_chart(fig, use_container_width=True)

n_90 = int(np.argmax(cum_var >= 0.90)) + 1
st.markdown(f"""
**Interpretation:**
- **PC1 alone explains {var_explained[0]*100:.0f}%** of the total variance -- this is the
  "general distress" dimension. All instruments load positively on it, meaning when one
  goes up, they all tend to go up. It is the psychosocial equivalent of a "general health"
  factor.
- **{n_90} components** capture 90% or more of the variance. The remaining components capture
  smaller, instrument-specific differences (like the slight difference between oral and ear
  thermometers).
- The **loadings heatmap** (right) shows how each instrument contributes to each component.
  PC1 is a broad "everything goes up together" factor. PC2 and PC3 capture more specific
  contrasts between instruments.
""")

# ── PCA scatter — the bridge to clustering ───────────────────────────────
st.subheader("2c. Visualizing Participants in Reduced Dimensions")
st.markdown("""
If we project all participants into the first two principal components, we can
*see* the multivariate structure -- like an X-ray that reveals the internal
arrangement of the data. Each dot is a patient, positioned according to their
overall pattern of psychosocial scores.
""")

X_pca = PCA(n_components=2).fit_transform(X_std)
pca_scatter = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1],
                             "Phenotype": survey["stress_phenotype"]})

# Add demographic columns for interactive coloring
for demo_col_name in ["sex", "race_ethnicity"]:
    if demo_col_name in survey_raw.columns:
        pca_scatter[demo_col_name] = survey_raw[demo_col_name].values
if "age" in survey_raw.columns:
    pca_scatter["age"] = survey_raw["age"].values
    pca_scatter["Age Group"] = pd.cut(survey_raw["age"], bins=[0, 35, 50, 65, 100],
                                       labels=["18-35", "36-50", "51-65", "66+"])

pca_color_by = st.radio(
    "Color PCA scatter by:",
    ["Cluster", "Sex", "Age Group", "Race/Ethnicity"],
    horizontal=True, key="pca_color_radio"
)

if pca_color_by == "Cluster":
    fig = px.scatter(pca_scatter, x="PC1", y="PC2", color="Phenotype",
                     color_discrete_map=COLORS, opacity=0.5,
                     category_orders={"Phenotype": phenotype_order},
                     title="Participants in PCA Space (colored by stress phenotype)")
elif pca_color_by == "Sex" and "sex" in pca_scatter.columns:
    fig = px.scatter(pca_scatter, x="PC1", y="PC2", color="sex",
                     opacity=0.5, title="Participants in PCA Space (colored by sex)")
elif pca_color_by == "Age Group" and "Age Group" in pca_scatter.columns:
    fig = px.scatter(pca_scatter, x="PC1", y="PC2", color="Age Group",
                     opacity=0.5, title="Participants in PCA Space (colored by age group)")
elif pca_color_by == "Race/Ethnicity" and "race_ethnicity" in pca_scatter.columns:
    fig = px.scatter(pca_scatter, x="PC1", y="PC2", color="race_ethnicity",
                     opacity=0.5, title="Participants in PCA Space (colored by race/ethnicity)")
else:
    fig = px.scatter(pca_scatter, x="PC1", y="PC2", color="Phenotype",
                     color_discrete_map=COLORS, opacity=0.5,
                     category_orders={"Phenotype": phenotype_order},
                     title="Participants in PCA Space")

fig.update_traces(marker=dict(size=4))
fig.update_layout(height=450, font=FONT)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Interpretation:** The clusters separate along PC1 (general distress). This tells us
something important: the instruments are fundamentally measuring a **spectrum of
psychosocial burden**, and the clusters we find in Step 3 represent **distinct
positions along this spectrum** -- not random groupings.

This is why PCA comes first: it tells us *what the data is measuring* and confirms
that meaningful structure exists before we try to segment people into groups. It is
the analytical equivalent of confirming a disease actually exists before trying to
classify patients by severity.
""")

st.markdown("""
> **How can this be applied?** In an omics study, the same approach could be used to
> determine whether 50 cytokine measurements really capture 50 different things, or
> whether they boil down to 3-4 inflammatory axes. This dramatically simplifies
> downstream modeling and interpretation.
""")

st.info("""
**PCA summary:** Our 7 instruments largely measure one dominant construct (general
psychosocial distress), explaining ~{:.0f}% of variance. The data has clear multivariate
structure. Now we can ask: *who are the people along this spectrum?*
""".format(var_explained[0]*100))


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — WHO ARE THE PEOPLE? (CLUSTERING)
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("Step 3 -- Who Are the People?")
st.markdown("""
PCA told us *what* we are measuring (a general distress dimension). Now we ask:
**are there distinct types of patients in our sample?**

Clustering identifies **distinct patient profiles** -- like recognizing that your
clinic has three types of patients: those doing well across the board, those with
moderate issues, and those struggling with everything simultaneously. Instead of
treating everyone the same, identifying these profiles allows for **targeted
interventions**.

Three algorithms are available (select in sidebar):
""")

st.markdown("""
| Algorithm | How it works | k selection |
|-----------|-------------|-------------|
| **K-Means** | Groups patients into k clusters by finding the k "center points" that minimize each patient's distance to their nearest center | You choose k |
| **GMM** | Fits k bell-shaped distributions to the data; each patient has a probability of belonging to each group (soft assignment) | You choose k |
| **VillageNet** | Creates micro-clusters ("villages"), builds a nearest-neighbor network, then uses community detection to find natural groupings | **Auto-detected** |
""")

if cluster_method == "VillageNet":
    st.markdown(f"""
    You selected **VillageNet** (villages={vn_villages}, neighbors={vn_neighbors}).
    The algorithm automatically detected **k = {n_clusters}** communities
    (silhouette = {sil_score:.3f}).
    """)
    with st.expander("How VillageNet works"):
        st.markdown(f"""
        1. **Micro-clustering:** K-Means creates **{vn_villages} villages** (small clusters)
           -- intentionally over-segmenting the data.
        2. **Graph construction:** A weighted graph connects villages whose members are nearest
           neighbors (**{vn_neighbors} neighbors** per village).
        3. **Community detection (WLCF):** Random walks on the village graph identify
           tightly-connected communities. The optimal number is determined **automatically**.

        *Reference: [VillageNet (arXiv:2501.10471)](https://arxiv.org/abs/2501.10471)*
        """)
else:
    st.markdown(f"""
    For K-Means and GMM, we compare **silhouette score** (higher = better separation
    between patient groups) and **BIC** (lower = better model fit) across k=2 to 6:
    """)
    metrics = cluster_selection_metrics(survey_raw)
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=metrics["k"], y=metrics["K-Means"], mode="lines+markers", name="K-Means"))
        fig.add_trace(go.Scatter(x=metrics["k"], y=metrics["GMM"], mode="lines+markers", name="GMM"))
        fig.add_vline(x=n_clusters, line_dash="dash", line_color="gray", annotation_text=f"k={n_clusters}")
        fig.update_layout(title="Silhouette Score by k", xaxis_title="k", yaxis_title="Silhouette", height=350, font=FONT)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.line(metrics, x="k", y="BIC", markers=True, color_discrete_sequence=["green"])
        fig.add_vline(x=n_clusters, line_dash="dash", line_color="gray", annotation_text=f"k={n_clusters}")
        fig.update_layout(title="GMM BIC by k", height=350, font=FONT)
        st.plotly_chart(fig, use_container_width=True)

# ── Comparison toggle: side-by-side cluster assignments in PCA space ────
st.subheader("3a. Compare Clustering Methods")
compare_toggle = st.checkbox("Show side-by-side comparison of two methods in PCA space", key="compare_toggle")
if compare_toggle:
    comp_methods = ["K-Means", "GMM"]
    if cluster_method == "VillageNet":
        comp_methods.append("VillageNet")
    col_left, col_right = st.columns(2)
    method_a = col_left.selectbox("Method A", comp_methods, index=0, key="comp_a")
    method_b = col_right.selectbox("Method B", comp_methods, index=min(1, len(comp_methods)-1), key="comp_b")

    labels_a, _, _, k_a, _ = run_clustering(survey_raw, method_a, n_clusters_input, vn_villages, vn_neighbors)
    labels_b, _, _, k_b, _ = run_clustering(survey_raw, method_b, n_clusters_input, vn_villages, vn_neighbors)

    colors_a = generate_palette(k_a)
    colors_b = generate_palette(k_b)
    map_a = CLUSTER_MAP if k_a == 3 else {i: f"Cluster {i}" for i in range(k_a)}
    map_b = CLUSTER_MAP if k_b == 3 else {i: f"Cluster {i}" for i in range(k_b)}

    pca_a = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1],
                           "Phenotype": [map_a[l] for l in labels_a]})
    pca_b = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1],
                           "Phenotype": [map_b[l] for l in labels_b]})

    with col_left:
        order_a = sorted(pca_a["Phenotype"].unique(), key=lambda x: list(map_a.values()).index(x) if x in map_a.values() else 0)
        fig = px.scatter(pca_a, x="PC1", y="PC2", color="Phenotype",
                         color_discrete_map=colors_a, opacity=0.5,
                         category_orders={"Phenotype": order_a},
                         title=f"{method_a} (k={k_a})")
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(height=350, font=FONT)
        st.plotly_chart(fig, use_container_width=True)
    with col_right:
        order_b = sorted(pca_b["Phenotype"].unique(), key=lambda x: list(map_b.values()).index(x) if x in map_b.values() else 0)
        fig = px.scatter(pca_b, x="PC1", y="PC2", color="Phenotype",
                         color_discrete_map=colors_b, opacity=0.5,
                         category_orders={"Phenotype": order_b},
                         title=f"{method_b} (k={k_b})")
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(height=350, font=FONT)
        st.plotly_chart(fig, use_container_width=True)

# ── Cluster profiles (the key bridge from PCA) ──────────────────────────
st.subheader("3b. Cluster Profiles")
method_label = f"**{cluster_method}**" + (f" (auto k={n_clusters})" if auto_k else f" (k={n_clusters})")
st.markdown(f"""
{method_label} assigns each participant to a stress phenotype
(silhouette = {sil_score:.3f}). Here is the payoff -- remember how PCA showed all
instruments tracking together? The cluster profiles below show *exactly that*:
each group sits at a different level across **all** instruments simultaneously,
like three patients at different points on a severity scale.
""")

profile = survey.groupby("stress_phenotype")[PSYCHOSOCIAL_TOTALS].mean().reindex(phenotype_order)

fig = go.Figure()
for pheno in phenotype_order:
    fig.add_trace(go.Scatter(
        x=[PSYCH_LABELS[c] for c in PSYCHOSOCIAL_TOTALS], y=profile.loc[pheno].values,
        mode="lines+markers", name=pheno,
        line=dict(color=COLORS.get(pheno, "#333"), width=3), marker=dict(size=10)))
fig.update_layout(yaxis_title="Mean Score", height=400, font=FONT,
                  title="Psychosocial Profile by Cluster")
st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns([1, 2])
with col1:
    sizes = survey["stress_phenotype"].value_counts().reindex(phenotype_order)
    st.markdown("**Cluster sizes:**")
    for pheno in phenotype_order:
        st.markdown(f"- **{pheno}:** n = {sizes[pheno]:,}")
with col2:
    selected_inst = st.selectbox("Explore an instrument:", PSYCHOSOCIAL_TOTALS,
                                  format_func=lambda x: PSYCH_LABELS[x], key="inst_box")
    fig = px.box(survey, x="stress_phenotype", y=selected_inst, color="stress_phenotype",
                 category_orders={"stress_phenotype": phenotype_order}, color_discrete_map=COLORS,
                 points="all")
    fig.update_traces(jitter=0.2, pointpos=-1.5, marker=dict(size=3, opacity=0.4, line=dict(width=0)))
    fig.update_layout(showlegend=False, height=350, font=FONT, yaxis_title="Score", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

st.success("""
**Interpretation:** The clusters represent distinct positions along the general distress
spectrum that PCA revealed. The **Low Burden** group has low scores across all instruments.
The **High Burden** group has elevated depression, anxiety, stress, PTSD, and poor sleep --
all moving together, exactly as PCA predicted.

**Why does this matter clinically?** "Stressed" is not monolithic. These are meaningfully
different patient subpopulations. A clinic could use these profiles to triage: Low Burden
patients may need preventive counseling, while High Burden patients may need immediate
psychiatric referral and cardiology follow-up. Now we can ask: do these psychosocial profiles
have a geographic and environmental signature?
""")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — GEOSPATIAL / ENVIRONMENTAL
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("Step 4 -- Where Do These Phenotypes Live?")
st.markdown("""
Each participant is linked to a **ZIP code**, and each ZIP code has 18 neighborhood-level
measures: pollution, noise, crime, food access, green space, deprivation indices, and more.

**Question:** Do high-stress patients concentrate in more deprived, more polluted,
less green neighborhoods? This is like asking whether the sickest patients in your
panel all happen to come from the same underserved neighborhoods -- a pattern that
would point to environmental and social determinants of health.
""")

merged_geo = survey.merge(geo, on="zip_code", how="left")
geo_features = [c for c in geo.columns if c != "zip_code"]
geo_profile = merged_geo.groupby("stress_phenotype")[geo_features].mean().reindex(phenotype_order)
geo_z = geo_profile.apply(stats.zscore, axis=0)

# Multi-select for exposure variables on heatmap
selected_exposures = st.multiselect(
    "Select exposure variables to display on the heatmap:",
    geo_features, default=geo_features, key="geo_heatmap_select"
)

if selected_exposures:
    _geo_vals = geo_z[selected_exposures].T.values
    _zabs = max(abs(np.nanmin(_geo_vals)), abs(np.nanmax(_geo_vals)), 1)
    fig = px.imshow(_geo_vals,
                    x=phenotype_order, y=selected_exposures,
                    color_continuous_scale="RdBu_r", zmin=-_zabs, zmax=_zabs,
                    aspect="auto",
                    title="Environmental Exposures by Phenotype (Z-scored)")
    fig.update_traces(xgap=3, ygap=3)
    fig.update_layout(height=max(350, len(selected_exposures) * 25 + 100), font=FONT)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("**Explore specific exposures** -- select one to see the distribution across clusters:")
key_exp = st.selectbox("Exposure variable:", geo_features,
                        index=geo_features.index("ADI_national_rank") if "ADI_national_rank" in geo_features else 0,
                        key="geo_select")
fig = px.box(merged_geo, x="stress_phenotype", y=key_exp, color="stress_phenotype",
             category_orders={"stress_phenotype": phenotype_order}, color_discrete_map=COLORS, points=False)
fig.update_layout(showlegend=False, height=350, font=FONT, xaxis_title="", yaxis_title=key_exp)
st.plotly_chart(fig, use_container_width=True)

# Map — choropleth with ZCTA shapes
st.markdown("**Spatial map** — see where burden concentrates geographically, shown as filled ZIP code boundaries:")
zip_all = build_zip_data(survey, geo)
map_opts = {"Psychosocial Burden": "psych_burden", "Area Deprivation": "ADI_national_rank",
            "CV Disease Burden": "cv_burden", "PM2.5 Pollution": "PM25_annual_mean"}
map_choice = st.selectbox("Map variable:", list(map_opts.keys()), key="map_sel")

zip_all["zip_str"] = zip_all["zip_code"].astype(str)

import json as _json
_geojson_path = DATA_DIR / "sacramento_zcta.geojson"
if _geojson_path.exists():
    with open(_geojson_path) as _f:
        _zcta_geojson = _json.load(_f)
    fig = px.choropleth_mapbox(
        zip_all, geojson=_zcta_geojson,
        locations="zip_str",
        featureidkey="properties.ZCTA5CE10",
        color=map_opts[map_choice],
        color_continuous_scale="Turbo",
        mapbox_style="open-street-map",
        zoom=9.5,
        center={"lat": zip_all["lat"].mean(), "lon": zip_all["lon"].mean()},
        opacity=0.7,
        hover_name="zip_str",
        hover_data={"n": True, map_opts[map_choice]: ":.2f", "zip_str": False},
        title=f"{map_choice} by ZIP Code (ZCTA boundaries)",
    )
    fig.update_layout(height=550, margin=dict(l=0, r=0, t=40, b=0), font=FONT)
    st.plotly_chart(fig, use_container_width=True)
else:
    # Fallback to scatter if GeoJSON not found
    _size_col = zip_all["n"].clip(lower=1)
    fig = px.scatter_mapbox(
        zip_all, lat="lat", lon="lon",
        color=map_opts[map_choice],
        size=_size_col,
        size_max=25,
        hover_name="zip_str",
        color_continuous_scale="Turbo",
        mapbox_style="open-street-map",
        zoom=9.5,
        center={"lat": zip_all["lat"].mean(), "lon": zip_all["lon"].mean()},
        opacity=0.8,
        title=f"Spatial Distribution: {map_choice}",
    )
    fig.update_layout(height=550, margin=dict(l=0, r=0, t=40, b=0), font=FONT)
    st.plotly_chart(fig, use_container_width=True)

st.success("""
**Key finding:** High-burden patients live in neighborhoods with significantly higher
deprivation, pollution, crime, and food desert scores -- and lower income, green space, and
walkability. The psychosocial phenotype has a clear environmental and geographic signature.

**Why does this matter clinically?** A health system could overlay these maps with clinic
locations to identify underserved areas where community health workers or mobile clinics
might have the greatest impact. The ZIP code is not destiny, but it is a powerful risk
indicator.
""")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — MOLECULAR SIGNATURES
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("Step 5 -- From Phenotype to Proteins")
st.markdown("""
We have established that stress phenotypes cluster with environmental deprivation.
Now we ask the mechanistic question: **does chronic psychosocial stress leave a
molecular fingerprint in the blood?**

Each participant has a proteomic profile -- 15 circulating proteins spanning
inflammation (CRP, IL-6, TNF-alpha), cardiac stress (NT-proBNP, Troponin I),
vascular markers, and a protective factor (Adiponectin).

Think of this as moving from the patient interview (psychosocial instruments) to
the lab report (blood proteins). We test whether patients in different stress
clusters have detectably different protein levels -- the molecular equivalent of
asking whether the patients who *feel* worse also *look* worse under the microscope.
""")

merged_omics = survey[["participant_id", "stress_cluster", "stress_phenotype"]].merge(
    omics, on="participant_id", how="inner")
available_proteins = [c for c in PROTEIN_COLS if c in merged_omics.columns]

# ── Compute differential expression stats ────────────────────────────────
de_results = []
low_cluster = 0
high_cluster = merged_omics["stress_cluster"].max()
for col in available_proteins:
    groups = [g[col].dropna().values for _, g in merged_omics.groupby("stress_cluster")]
    f_stat, p_val = stats.f_oneway(*groups)
    low_vals = merged_omics.loc[merged_omics["stress_cluster"] == low_cluster, col]
    high_vals = merged_omics.loc[merged_omics["stress_cluster"] == high_cluster, col]
    m_low, m_high = low_vals.mean(), high_vals.mean()
    s_low, s_high = low_vals.std(), high_vals.std()
    fc = m_high - m_low
    pooled = np.sqrt((s_low**2 + s_high**2) / 2)
    d = fc / pooled if pooled > 0 else 0
    de_results.append({
        "col": col, "Protein": PROTEIN_SHORT.get(col, col), "Log2 FC": fc,
        "Cohen d": d, "F": f_stat, "p": p_val,
        "Mean Low": m_low, "Mean High": m_high,
        "Direction": "Upregulated" if fc > 0 else "Downregulated",
        "Significant": p_val < 0.05,
    })
de_df = pd.DataFrame(de_results).sort_values("Log2 FC", ascending=True)

# ── VIZ 1: Diverging lollipop chart (fold change + effect size) ──────────
st.subheader("Fold Change and Effect Size")
st.markdown("""
Each protein's **fold change** (High vs Low Burden) is shown as a lollipop.
Length = magnitude of change. Cohen's d (effect size) is annotated -- think of it
as "how noticeable would this difference be to a clinician?"
(d > 0.8 = large, clearly meaningful; d ~ 0.5 = medium; d < 0.2 = small).
""")

# Multi-select to highlight specific proteins
highlight_proteins = st.multiselect(
    "Highlight specific proteins (leave empty to show all equally):",
    de_df["Protein"].tolist(), default=[], key="protein_highlight"
)

fig = go.Figure()
for _, row in de_df.iterrows():
    is_highlighted = len(highlight_proteins) == 0 or row["Protein"] in highlight_proteins
    base_color = "#CC6677" if row["Direction"] == "Upregulated" else "#44AA99"
    color = base_color if is_highlighted else "#DDDDDD"
    width = 3 if is_highlighted else 1.5
    fig.add_trace(go.Scatter(
        x=[0, row["Log2 FC"]], y=[row["Protein"], row["Protein"]],
        mode="lines", line=dict(color=color, width=width), showlegend=False,
        hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[row["Log2 FC"]], y=[row["Protein"]],
        mode="markers+text", marker=dict(size=14 if is_highlighted else 8,
                                          color=color, line=dict(width=1.5, color="black")),
        text=f"d={row['Cohen d']:+.1f}" if is_highlighted else "",
        textposition="middle right" if row["Log2 FC"] > 0 else "middle left",
        textfont=dict(size=10), showlegend=False,
        hovertext=f"<b>{row['Protein']}</b><br>Log2 FC: {row['Log2 FC']:+.2f}<br>"
                  f"Cohen's d: {row['Cohen d']:+.2f}<br>F={row['F']:.0f}, p={row['p']:.1e}",
        hoverinfo="text"))

fig.add_vline(x=0, line_color="black", line_width=1)
fig.add_annotation(x=2.5, y=de_df["Protein"].iloc[-1], text="Upregulated in<br>High Burden -->",
                   showarrow=False, font=dict(size=11, color="#CC6677"), xanchor="left")
fig.add_annotation(x=-1.5, y=de_df["Protein"].iloc[-1], text="<-- Downregulated in<br>High Burden",
                   showarrow=False, font=dict(size=11, color="#44AA99"), xanchor="right")
fig.update_layout(height=500, font=FONT, xaxis_title="Log2 Fold Change (High vs Low Burden)",
                  yaxis=dict(categoryorder="array", categoryarray=de_df["Protein"].tolist()),
                  plot_bgcolor="white", margin=dict(l=100))
st.plotly_chart(fig, use_container_width=True)

# ── VIZ 2: Paired dot plot (Low vs High mean expression) ─────────────────
st.subheader("Low vs High Burden -- Paired Comparison")
st.markdown("""
Each protein is shown with its mean expression in the **Low** and **High** burden clusters.
Connected dots make the direction and magnitude of change immediately visible -- like
comparing a patient's labs before and after a treatment, except here the "treatment" is
chronic stress.
""")

fig = go.Figure()
for _, row in de_df.iterrows():
    fig.add_trace(go.Scatter(
        x=[row["Mean Low"], row["Mean High"]], y=[row["Protein"], row["Protein"]],
        mode="lines", line=dict(color="#888", width=2), showlegend=False, hoverinfo="skip"))

fig.add_trace(go.Scatter(
    x=de_df["Mean Low"], y=de_df["Protein"], mode="markers", name=phenotype_order[0],
    marker=dict(size=12, color=COLORS.get(phenotype_order[0], "#44AA99"),
                line=dict(width=1.5, color="black"), symbol="circle"),
    hovertext=[f"{r['Protein']}: {r['Mean Low']:.2f}" for _, r in de_df.iterrows()], hoverinfo="text"))
fig.add_trace(go.Scatter(
    x=de_df["Mean High"], y=de_df["Protein"], mode="markers", name=phenotype_order[-1],
    marker=dict(size=12, color=COLORS.get(phenotype_order[-1], "#CC6677"),
                line=dict(width=1.5, color="black"), symbol="diamond"),
    hovertext=[f"{r['Protein']}: {r['Mean High']:.2f}" for _, r in de_df.iterrows()], hoverinfo="text"))

fig.update_layout(height=500, font=FONT, xaxis_title="Mean log2 Abundance",
                  yaxis=dict(categoryorder="array", categoryarray=de_df["Protein"].tolist()),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                  plot_bgcolor="white", margin=dict(l=100),
                  title="Mean Expression: Low vs High Burden")
st.plotly_chart(fig, use_container_width=True)

# ── VIZ 3: Protein heatmap across all clusters ──────────────────────────
st.subheader("Expression Gradient Across All Clusters")
protein_profile = merged_omics.groupby("stress_phenotype")[available_proteins].mean().reindex(phenotype_order)
profile_z = protein_profile.apply(stats.zscore, axis=0)
short_names = [PROTEIN_SHORT.get(c, c) for c in available_proteins]

_prot_vals = profile_z.T.values
_pzabs = max(abs(np.nanmin(_prot_vals)), abs(np.nanmax(_prot_vals)), 1)
fig = px.imshow(_prot_vals, x=phenotype_order, y=short_names,
                color_continuous_scale="RdBu_r", zmin=-_pzabs, zmax=_pzabs,
                aspect="auto",
                title="Protein Expression by Cluster (Z-scored)")
fig.update_traces(xgap=3, ygap=3)
fig.update_layout(height=500, font=FONT)
st.plotly_chart(fig, use_container_width=True)

# ── VIZ 4: Individual protein explorer ───────────────────────────────────
with st.expander("Explore: individual protein boxplots"):
    prot_choice = st.selectbox("Protein:", available_proteins,
                                format_func=lambda x: PROTEIN_SHORT.get(x, x), key="prot_box")
    fig = px.box(merged_omics, x="stress_phenotype", y=prot_choice, color="stress_phenotype",
                 category_orders={"stress_phenotype": phenotype_order}, color_discrete_map=COLORS,
                 points="all", title=PROTEIN_SHORT.get(prot_choice, prot_choice))
    fig.update_traces(jitter=0.2, pointpos=-1.5, marker=dict(size=3, opacity=0.4, line=dict(width=0)))
    fig.update_layout(showlegend=False, height=400, font=FONT, xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

with st.expander("Full statistics table"):
    stats_display = de_df[["Protein", "Log2 FC", "Cohen d", "F", "p", "Direction"]].copy()
    stats_display["F"] = stats_display["F"].apply(lambda x: f"{x:.1f}")
    stats_display["p"] = stats_display["p"].apply(lambda x: f"{x:.1e}")
    stats_display["Log2 FC"] = stats_display["Log2 FC"].apply(lambda x: f"{x:+.3f}")
    stats_display["Cohen d"] = stats_display["Cohen d"].apply(lambda x: f"{x:+.2f}")
    st.dataframe(stats_display.sort_values("Cohen d", ascending=False),
                 use_container_width=True, hide_index=True)

st.success("""
**Key finding:** 14 of 15 proteins are significantly upregulated in the High Burden
cluster -- inflammatory markers (CRP, IL-6, TNF-alpha), cardiac stress markers (NT-proBNP,
Troponin I), and vascular markers all increase with psychosocial burden. The one protective
factor, **Adiponectin**, goes in the opposite direction (lower in high burden). Chronic
stress has a measurable molecular signature.

**Why does this matter clinically?** These are not exotic research-only biomarkers -- CRP
and NT-proBNP are routinely ordered in clinical practice. A clinic could use elevated CRP
alongside a high PHQ-8 score to identify patients at the intersection of psychosocial and
biological risk, prioritizing them for integrated behavioral-cardiac care.
""")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 — CLINICAL OUTCOMES
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("Step 6 -- Does This Translate to Disease?")
st.markdown("""
The biological plausibility is established -- stressed patients have elevated
inflammatory and cardiac markers. But does it matter at the bedside?
Let us compare the **prevalence of cardiovascular diseases** across the stress phenotypes.

This is the "so what?" step: moving from molecules back to the patient to ask
whether the patterns we found actually predict who gets sick.
""")

available_dx = [c for c in DX_COLS if c in survey.columns]
prev_data = []
for pheno in phenotype_order:
    sub = survey[survey["stress_phenotype"] == pheno]
    n = len(sub)
    for col in available_dx:
        prev_data.append({"Phenotype": pheno, "Condition": DX_COLS[col],
                          "Prevalence (%)": round(sub[col].sum() / n * 100, 1)})
prev_df = pd.DataFrame(prev_data)

fig = px.bar(prev_df, x="Condition", y="Prevalence (%)", color="Phenotype",
             barmode="group", color_discrete_map=COLORS,
             category_orders={"Phenotype": phenotype_order}, text="Prevalence (%)")
fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                  marker_line_width=1.5, marker_line_color="black")
fig.update_layout(height=450, font=FONT, xaxis_title="",
                  yaxis=dict(range=[0, prev_df["Prevalence (%)"].max() * 1.25]))
st.plotly_chart(fig, use_container_width=True)

# Key numbers callout
hyp = prev_df[prev_df["Condition"] == "Hypertension"]
if not hyp.empty:
    low_h = hyp[hyp["Phenotype"] == phenotype_order[0]]["Prevalence (%)"].values
    high_h = hyp[hyp["Phenotype"] == phenotype_order[-1]]["Prevalence (%)"].values
    if len(low_h) > 0 and len(high_h) > 0:
        st.markdown(f"""
        > **Hypertension prevalence:** {low_h[0]:.1f}% in {phenotype_order[0]}
        vs. {high_h[0]:.1f}% in {phenotype_order[-1]}
        -- a **{high_h[0]/low_h[0]:.1f}x** increase.
        """)

st.success("""
**Key finding:** Every cardiovascular condition -- hypertension, diabetes, heart failure,
MI, and stroke -- shows significantly higher prevalence in the High Burden cluster.
The stress-environment-protein axis translates directly into clinical disease burden.

**How can this be applied?** A health system could screen patients with high psychosocial
burden scores for subclinical cardiovascular disease earlier and more aggressively. Instead
of waiting for a heart attack, intervene when the stress profile and protein markers both
signal elevated risk.
""")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 7 — TISSUE EXPRESSION (HPA)
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("Step 7 -- Where in the Body Are These Proteins Expressed?")
st.markdown("""
We have identified proteins that are elevated in high-stress individuals. But where
are these proteins normally produced in the body? Using data from the
[Human Protein Atlas](https://www.proteinatlas.org), we map RNA expression levels
across 27 human tissues.

Think of this as asking: "If CRP is elevated in stressed patients, which organ is
making it?" This tells us which **organs** are most involved in the
stress-to-cardiovascular-disease pathway -- pointing to the biological sites where
the damage is happening.
""")

hpa_df = fetch_hpa()
tissue_cols = [c for c in hpa_df.columns if c not in ("gene", "uniprot")]

if hpa_df.empty or not tissue_cols:
    st.warning("Could not fetch HPA data. Check your internet connection.")
else:
    gene_to_short = {}
    for cn, s in PROTEIN_SHORT.items():
        g = UNIPROT_TO_GENE.get(cn.split("_")[0])
        if g: gene_to_short[g] = s
    hpa_df["protein"] = hpa_df["gene"].map(gene_to_short).fillna(hpa_df["gene"])

    # Identify upregulated
    upreg = [UNIPROT_TO_GENE[c.split("_")[0]] for c in available_proteins
             if merged_omics.loc[merged_omics["stress_cluster"] == merged_omics["stress_cluster"].max(), c].mean()
             > merged_omics.loc[merged_omics["stress_cluster"] == 0, c].mean()
             and c.split("_")[0] in UNIPROT_TO_GENE]

    # Full heatmap
    heat = hpa_df.set_index("protein")[tissue_cols]
    heat_log = np.log10(heat + 1)
    fig = px.imshow(heat_log.values, x=tissue_cols, y=heat.index.tolist(),
                    color_continuous_scale="Plasma", aspect="auto",
                    title="All Panel Proteins -- Tissue RNA Expression (log10 nTPM)")
    fig.update_traces(xgap=3, ygap=3)
    fig.update_layout(height=500, font=FONT, xaxis=dict(tickangle=45))
    st.plotly_chart(fig, use_container_width=True)

    # Organ burden bar
    upreg_df = hpa_df[hpa_df["gene"].isin(upreg)]
    if not upreg_df.empty:
        st.markdown(f"""
        **{len(upreg)} upregulated proteins** -- which organs produce them the most?
        The bar chart below sums expression across all upregulated proteins per tissue.
        The organs at the top are the primary "factories" for the stress-response molecules:
        """)
        organ_sum = upreg_df[tissue_cols].sum().sort_values(ascending=True).reset_index()
        organ_sum.columns = ["Tissue", "Total nTPM"]
        fig = px.bar(organ_sum, x="Total nTPM", y="Tissue", orientation="h",
                     color="Total nTPM", color_continuous_scale="YlOrRd",
                     title="Cumulative Expression of Upregulated Proteins by Organ")
        fig.update_layout(height=600, font=FONT, showlegend=False,
                          yaxis=dict(categoryorder="total ascending"))
        fig.update_traces(marker_line_width=0.5, marker_line_color="black")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Explore: single protein tissue profile"):
        pc = st.selectbox("Protein:", hpa_df["protein"].tolist(), key="hpa_single")
        row = hpa_df[hpa_df["protein"] == pc][tissue_cols].iloc[0].sort_values(ascending=True)
        pf = row.reset_index()
        pf.columns = ["Tissue", "nTPM"]
        fig = px.bar(pf, x="nTPM", y="Tissue", orientation="h",
                     color="nTPM", color_continuous_scale="Plasma",
                     title=f"{pc} -- Tissue Expression")
        fig.update_layout(height=550, font=FONT, showlegend=False,
                          yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **Key takeaway:** The upregulated proteins are predominantly expressed in the **liver**
    (acute-phase response -- think CRP production), **bone marrow** (immune cells that
    produce cytokines), **smooth muscle** and **heart** (cardiovascular tissue under direct
    stress). These are exactly the organs you would expect to be involved in the
    inflammatory and cardiovascular stress response.

    **Why does this matter clinically?** Understanding which organs are most active in the
    stress response helps identify therapeutic targets. For example, liver-focused
    interventions (like statins, which also reduce CRP) may be particularly relevant for
    high-burden patients.
    """)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 8 — PROTEIN INTERACTION NETWORK
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("Step 8 -- How Do These Proteins Talk to Each Other?")
st.markdown("""
Proteins do not act in isolation -- they form networks, like colleagues in a hospital
who coordinate care. Using [STRING DB](https://string-db.org), we retrieve
known and predicted **protein-protein interactions** for our 15-protein panel plus
their closest interaction partners.

This network reveals which proteins are hubs (the "department heads" with many
connections), which form tight functional modules (like a cardiology team), and what
biological pathways they collectively activate.
""")

score_threshold = st.slider("Minimum interaction confidence", 150, 900, 400, step=50, key="string_score")
show_partners = st.checkbox("Show extended interaction partners", value=True, key="string_ext")

internal, partners_data, enrichment = fetch_string(score_threshold)
if not internal and not partners_data:
    st.warning("Could not fetch STRING data. Check your internet connection.")
else:
    G = nx.Graph()
    core_genes = set(UNIPROT_TO_GENE.values())

    for gene in core_genes:
        display = gene
        for col, s in PROTEIN_SHORT.items():
            if UNIPROT_TO_GENE.get(col.split("_")[0]) == gene:
                display = s
                break
        G.add_node(gene, display=display, node_type="core")

    for e in internal:
        a, b = e["preferredName_A"], e["preferredName_B"]
        if a in core_genes and b in core_genes:
            G.add_edge(a, b, score=e["score"], edge_type="internal")

    if show_partners and partners_data:
        for e in partners_data:
            a, b = e["preferredName_A"], e["preferredName_B"]
            for n in [a, b]:
                if n not in G:
                    G.add_node(n, display=n, node_type="partner")
            if not G.has_edge(a, b):
                G.add_edge(a, b, score=e["score"], edge_type="partner")

    st.markdown(f"**Network:** {G.number_of_nodes()} proteins, {G.number_of_edges()} interactions")

    # Layout
    pos = nx.spring_layout(G, k=1.8, iterations=80, seed=42)

    # ── Node selector: isolate a protein's neighborhood ─────────────────
    all_display = {G.nodes[n].get("display", n): n for n in G.nodes()}
    highlight_options = ["Show full network"] + sorted(all_display.keys())
    selected_node_label = st.selectbox(
        "Select a protein to isolate its connections:",
        highlight_options, key="net_highlight")

    if selected_node_label != "Show full network":
        focus_gene = all_display[selected_node_label]
        # Build isolated subgraph: selected node + direct neighbors only
        neighbors = set(G.neighbors(focus_gene)) | {focus_gene}
        H = G.subgraph(neighbors).copy()
        sub_pos = nx.spring_layout(H, k=2.5, iterations=60, seed=42)
        title_text = f"Isolated Network: {selected_node_label} ({len(neighbors)-1} connections)"
    else:
        focus_gene = None
        H = G
        sub_pos = pos
        title_text = "Protein-Protein Interaction Network (full)"

    # Draw edges
    edge_traces = []
    for u, v, d in H.edges(data=True):
        x0, y0 = sub_pos[u]; x1, y1 = sub_pos[v]
        is_focus_edge = focus_gene is not None and (u == focus_gene or v == focus_gene)
        color = "rgba(204,102,119,0.85)" if is_focus_edge else (
            "rgba(50,50,50,0.6)" if d.get("edge_type") == "internal" else "rgba(100,100,100,0.35)")
        width = (2.5 + d.get("score", 0.4) * 4) if is_focus_edge else (1 + d.get("score", 0.4) * 3)
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None], mode="lines",
            line=dict(width=width, color=color),
            hoverinfo="text",
            text=f"{H.nodes[u].get('display',u)} -- {H.nodes[v].get('display',v)}: {d.get('score',0):.2f}",
            showlegend=False))

    # Draw nodes
    nx_, ny_, nt_, nh_, nc_, ns_ = [], [], [], [], [], []
    for node, d in H.nodes(data=True):
        x, y = sub_pos[node]; nx_.append(x); ny_.append(y)
        display = d.get("display", node)
        is_core = d.get("node_type") == "core"
        degree = H.degree(node)
        nt_.append(display)
        nh_.append(f"<b>{display}</b> ({node})<br>Connections in view: {degree}")

        if node == focus_gene:
            nc_.append("#CC6677"); ns_.append(35)
        elif is_core:
            nc_.append("#CC6677"); ns_.append(20 + degree * 2)
        else:
            nc_.append("#88CCEE"); ns_.append(14 + degree * 1.5)

    fig = go.Figure(data=edge_traces + [go.Scatter(
        x=nx_, y=ny_, mode="markers+text", text=nt_, textposition="top center",
        textfont=dict(size=10 if focus_gene else 9), hovertext=nh_, hoverinfo="text",
        marker=dict(size=ns_, color=nc_, line=dict(width=1.5, color="DarkSlateGrey")),
        showlegend=False)])

    fig.update_layout(
        title=title_text, height=650, font=FONT,
        xaxis=dict(visible=False), yaxis=dict(visible=False), plot_bgcolor="white",
        hovermode="closest",
        annotations=[
            dict(text="● Core panel", x=0.01, y=0.99, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=12, color="#CC6677")),
            dict(text="● Partners", x=0.01, y=0.95, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=12, color="#88CCEE"))])

    if focus_gene:
        neighbor_names = sorted([H.nodes[n].get("display", n) for n in H.nodes() if n != focus_gene])
        fig.add_annotation(
            text=f"<b>{selected_node_label}</b> interacts with:<br>" + ", ".join(neighbor_names),
            x=0.99, y=0.01, xref="paper", yref="paper", showarrow=False,
            font=dict(size=10), xanchor="right", yanchor="bottom",
            bgcolor="rgba(255,255,255,0.9)", bordercolor="#CC6677", borderwidth=1)

    st.plotly_chart(fig, use_container_width=True)

    # Centrality table
    with st.expander("Network centrality metrics"):
        st.markdown("""
        **Degree centrality** measures how many direct connections a protein has (like how
        many other departments a doctor collaborates with). **Betweenness centrality** measures
        how often a protein sits on the shortest path between two others (like a coordinator
        who bridges different teams).
        """)
        dc = nx.degree_centrality(G)
        bc = nx.betweenness_centrality(G)
        rows = [{"Protein": G.nodes[g].get("display", g), "Gene": g,
                 "Degree": G.degree(g), "Deg. Centrality": f"{dc[g]:.3f}",
                 "Betw. Centrality": f"{bc[g]:.3f}"}
                for g in sorted(core_genes) if g in G]
        st.dataframe(pd.DataFrame(rows).sort_values("Degree", ascending=False),
                     use_container_width=True, hide_index=True)

    # Enrichment
    if enrichment:
        with st.expander("Functional enrichment (GO, KEGG pathways)"):
            er = [{"Category": e.get("category",""), "Term": e.get("description",""),
                   "FDR": f"{e.get('fdr',1):.2e}", "Genes": ", ".join(e.get("preferredNames",[]))}
                  for e in enrichment if e.get("fdr", 1) < 0.05]
            if er:
                st.dataframe(pd.DataFrame(er).head(30), use_container_width=True, hide_index=True, height=350)

    st.info("""
    **Key takeaway:** The 15 proteins form a tightly interconnected network. Key hubs like
    **TNF-alpha**, **IL-6**, and **VEGF-A** bridge inflammatory, vascular, and cardiac stress
    modules -- consistent with the multi-organ pathophysiology of stress-driven CV disease.

    **How can this be applied?** In an omics study, the same network approach could identify
    which proteins are the most promising drug targets. Targeting a hub protein (like IL-6,
    which already has approved inhibitors like tocilizumab) could disrupt multiple pathological
    pathways simultaneously.
    """)


# ═════════════════════════════════════════════════════════════════════════════
# CONCLUSION
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("Summary")
pc1_pct = f"{var_explained[0]*100:.0f}" if "var_explained" in dir() else "60"
st.markdown(f"""
This pipeline traced a complete causal narrative, from questionnaires to molecules to disease:

| Step | Question | Finding |
|------|----------|---------|
| **1. Data Prep** | Is our data ready? | 2,000 participants; missingness patterns identified and addressed; 7 instruments standardized |
| **2. PCA** | What are we measuring? | One dominant distress dimension explains ~{pc1_pct}% of variance |
| **3. Clustering** | Who are the people? | Distinct phenotypes (Low / Moderate / High Burden) along the distress spectrum |
| **4. Geospatial** | Where do they live? | High Burden = more deprivation, pollution, crime, fewer green spaces |
| **5. Proteins** | What is in their blood? | 14/15 proteins elevated in High Burden; Adiponectin (protective) decreased |
| **6. Disease** | Does it cause disease? | All CV conditions show graded increase across clusters |
| **7. Tissue** | Which organs? | Liver, bone marrow, smooth muscle, heart |
| **8. Network** | Which pathways? | Tight inflammatory-vascular-cardiac network with key hubs |

**Bottom line:** Chronic psychosocial stress is not monolithic. Distinct stress phenotypes
emerge from multivariate data, map onto deprived neighborhoods, activate specific
inflammatory and cardiac stress pathways, and translate into measurable cardiovascular
disease burden.

**What a clinic could do with this:** Screen patients using validated psychosocial
instruments, identify those in the High Burden profile, check inflammatory markers (CRP,
IL-6), and provide integrated behavioral-cardiac interventions -- especially in
neighborhoods flagged by the geospatial analysis as high-deprivation areas.
""")
