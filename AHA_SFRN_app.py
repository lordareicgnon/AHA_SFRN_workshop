import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import VillageNet as VN


def read_uploaded_csv(uploaded_file, transpose=False, has_headers=False, has_index=False):
    df = pd.read_csv(uploaded_file, header=None)

    if transpose:
        df = df.T
    if has_headers:
        df = df.rename(columns=df.iloc[0]).drop(df.index[0])
    if has_index:
        df = df.set_index(df.columns.tolist()[0])

    return df


def maybe_normalize(X, normalize):
    if not normalize:
        return X
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def run_villagenet(X, villages, neighbors, comms):
    model = VN.VillageNet(
        villages=int(villages),
        neighbors=int(neighbors),
        normalize=0,   # normalization is handled outside
    )
    if comms is not None:
        model.fit(X, comms=int(comms))
    else:
        model.fit(X)
    return np.asarray(model.comm_id)


def run_kmeans(X, n_clusters):
    model = KMeans(
        n_clusters=int(n_clusters),
        random_state=0,
        n_init=10
    )
    return model.fit_predict(X)


def run_gmm(X, n_clusters, covariance_type):
    model = GaussianMixture(
        n_components=int(n_clusters),
        covariance_type=covariance_type,
        random_state=0
    )
    return model.fit_predict(X)


def run_agglomerative(X, n_clusters, linkage):
    model = AgglomerativeClustering(
        n_clusters=int(n_clusters),
        linkage=linkage
    )
    return model.fit_predict(X)


def plot_clusters(X_pca, labels, method_name):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.set_title(f"{method_name} clustering")
    plt.tight_layout()
    return fig


st.set_page_config(page_title="Clustering App", layout="wide")

st.title("Clustering App")
st.write("Choose one clustering method, set its hyperparameters, and visualize results in PCA space.")

with st.expander("More Information"):
    st.write(
        """
        - Use the Wine dataset as sample data, or upload your own CSV.
        - Optionally normalize the data before clustering.
        - Choose one clustering algorithm from the dropdown.
        - Method-specific hyperparameters will appear automatically.
        - Results are visualized in PCA1 vs PCA2.
        """
    )

# ----------------------------
# Data input
# ----------------------------
st.subheader("Data")

use_sample_data = st.checkbox("Use Sample Data (Wine)", value=True)

uploaded_file = None
if not use_sample_data:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

data = None

if use_sample_data:
    wine = load_wine()
    data = wine["data"]
    df_display = pd.DataFrame(data, columns=wine["feature_names"])
    st.write("Using Wine sample data:")
    st.dataframe(df_display)
elif uploaded_file is not None:
    col1, col2, col3 = st.columns(3)
    transpose = col1.checkbox("Transpose Data", value=False)
    has_headers = col2.checkbox("Contains Headers", value=False)
    has_index = col3.checkbox("Contains Indices", value=False)

    try:
        df = read_uploaded_csv(
            uploaded_file,
            transpose=transpose,
            has_headers=has_headers,
            has_index=has_index
        )
        st.write("Uploaded data:")
        st.dataframe(df)
        data = np.asarray(df).astype(float)
    except Exception as e:
        st.error(f"Could not load the CSV as numeric data: {e}")

# ----------------------------
# Options and method selection
# ----------------------------
if data is not None:
    st.subheader("Options")
    normalize = st.checkbox("Normalize Data", value=False)
    X = maybe_normalize(data.copy(), normalize)

    st.subheader("Clustering Method")
    method = st.selectbox(
        "Choose clustering method",
        ["VillageNet", "K-Means", "Gaussian Mixture", "Agglomerative"]
    )

    st.subheader("Hyperparameters")

    # Method-specific hyperparameters
    vn_villages = None
    vn_neighbors = None
    vn_set_clusters = None
    vn_comms = None

    km_clusters = None

    gmm_clusters = None
    gmm_covariance = None

    agg_clusters = None
    agg_linkage = None

    if method == "VillageNet":
        col1, col2 = st.columns(2)

        vn_villages = col1.number_input(
            "Number of villages",
            min_value=2,
            max_value=int(X.shape[0]),
            value=int(min(200, X.shape[0])),
            step=1
        )

        vn_neighbors = col2.number_input(
            "Number of nearest neighbors",
            min_value=1,
            max_value=int(X.shape[0]),
            value=int(min(20, X.shape[0])),
            step=1
        )

        vn_set_clusters = st.checkbox("Set number of clusters manually", value=True)

        if vn_set_clusters:
            vn_comms = st.number_input(
                "Number of clusters",
                min_value=2,
                max_value=int(X.shape[0]),
                value=int(min(3, X.shape[0])),
                step=1
            )

    elif method == "K-Means":
        km_clusters = st.number_input(
            "Number of clusters",
            min_value=2,
            max_value=int(X.shape[0]),
            value=int(min(3, X.shape[0])),
            step=1
        )

    elif method == "Gaussian Mixture":
        col1, col2 = st.columns(2)

        gmm_clusters = col1.number_input(
            "Number of clusters",
            min_value=2,
            max_value=int(X.shape[0]),
            value=int(min(3, X.shape[0])),
            step=1
        )

        gmm_covariance = col2.selectbox(
            "Covariance type",
            ["full", "tied", "diag", "spherical"]
        )

    elif method == "Agglomerative":
        col1, col2 = st.columns(2)

        agg_clusters = col1.number_input(
            "Number of clusters",
            min_value=2,
            max_value=int(X.shape[0]),
            value=int(min(3, X.shape[0])),
            step=1
        )

        agg_linkage = col2.selectbox(
            "Linkage",
            ["ward", "complete", "average", "single"]
        )

    # ----------------------------
    # Run clustering
    # ----------------------------
    with st.form("cluster_form"):
        run = st.form_submit_button("Run Clustering")

    if run:
        try:
            if method == "VillageNet":
                labels = run_villagenet(
                    X,
                    villages=vn_villages,
                    neighbors=vn_neighbors,
                    comms=vn_comms if vn_set_clusters else None
                )

            elif method == "K-Means":
                labels = run_kmeans(X, km_clusters)

            elif method == "Gaussian Mixture":
                labels = run_gmm(X, gmm_clusters, gmm_covariance)

            elif method == "Agglomerative":
                labels = run_agglomerative(X, agg_clusters, agg_linkage)

            else:
                st.error("Unknown clustering method selected.")
                labels = None

            if labels is not None:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)

                st.subheader("PCA Visualization")
                fig = plot_clusters(X_pca, labels, method)
                st.pyplot(fig)

                st.subheader("Cluster Assignments")
                df_labels = pd.DataFrame({
                    "Index": np.arange(len(labels)),
                    "Cluster": labels
                })
                st.dataframe(df_labels)

                for cl in np.unique(labels):
                    inds = np.where(labels == cl)[0]
                    with st.expander(f"Cluster {cl}"):
                        st.write(", ".join(map(str, inds)))

        except Exception as e:
            st.error(f"{method} failed: {e}")
