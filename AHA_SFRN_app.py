import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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
        normalize=0,  # normalization handled outside
    )
    model.fit(X, comms=int(comms) if comms is not None else None)
    return np.asarray(model.comm_id)


def run_kmeans(X, comms):
    model = KMeans(
        n_clusters=int(comms),
        random_state=0,
        n_init=10,
    )
    return model.fit_predict(X)


def run_gmm(X, comms):
    model = GaussianMixture(
        n_components=int(comms),
        random_state=0,
    )
    return model.fit_predict(X)


def plot_pca_results(X_pca, results):
    n_methods = len(results)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))

    if n_methods == 1:
        axes = [axes]

    for ax, (method_name, labels) in zip(axes, results.items()):
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
        ax.set_title(method_name)
        ax.set_xlabel("PCA1")
        ax.set_ylabel("PCA2")

    plt.tight_layout()
    return fig


st.set_page_config(page_title="VillageNet Clustering App", layout="wide")

st.title("Multi-Method Clustering App")
st.write("Compare VillageNet, K-Means, and Gaussian Mixture Model on uploaded data or the Wine sample dataset.")

with st.expander("More Information"):
    st.write(
        """
        - Use the Wine dataset as sample data, or upload your own CSV.
        - Optionally normalize the data before clustering.
        - Compare VillageNet, K-Means, and Gaussian Mixture Model.
        - Visualize results in PCA1 vs PCA2.
        """
    )

st.subheader("Data")

use_sample_data = st.checkbox("Use Sample Data (Wine)", value=True)

uploaded_file = None
if not use_sample_data:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

data = None
display_df = None

if use_sample_data:
    wine = load_wine()
    data = wine["data"]
    display_df = pd.DataFrame(data, columns=wine["feature_names"])
    st.write("Using Wine sample data:")
    st.dataframe(display_df)
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
            has_index=has_index,
        )
        display_df = df
        st.write("Uploaded data:")
        st.dataframe(df)
        data = np.asarray(df).astype(float)
    except Exception as e:
        st.error(f"Could not load the CSV as numeric data: {e}")

if data is not None:
    st.subheader("Options")

    normalize = st.checkbox("Normalize Data", value=False)

    X = maybe_normalize(data.copy(), normalize)

    st.subheader("Methods")
    use_villagenet = st.checkbox("VillageNet", value=True)
    use_kmeans = st.checkbox("K-Means", value=True)
    use_gmm = st.checkbox("Gaussian Mixture Model", value=True)

    st.subheader("Hyperparameters")

    col1, col2 = st.columns(2)
    villages = col1.number_input(
        "Number of villages (VillageNet)",
        min_value=2,
        max_value=int(X.shape[0]),
        value=int(min(200, X.shape[0])),
        step=1,
    )
    neighbors = col2.number_input(
        "Number of nearest neighbors (VillageNet)",
        min_value=1,
        max_value=int(X.shape[0]),
        value=int(min(20, X.shape[0])),
        step=1,
    )

    set_num_clusters = st.checkbox("Set number of clusters manually", value=True)

    comms = None
    if set_num_clusters:
        comms = st.number_input(
            "Number of clusters",
            min_value=2,
            max_value=int(X.shape[0]),
            value=int(min(3, X.shape[0])),
            step=1,
        )

    with st.form("cluster_form"):
        run = st.form_submit_button("Run Clustering")

    if run:
        results = {}

        if use_villagenet:
            try:
                vn_labels = run_villagenet(X, villages, neighbors, comms)
                results["VillageNet"] = vn_labels
            except Exception as e:
                st.error(f"VillageNet failed: {e}")

        if use_kmeans:
            if comms is None:
                st.error("K-Means requires the number of clusters to be set.")
            else:
                try:
                    km_labels = run_kmeans(X, comms)
                    results["K-Means"] = km_labels
                except Exception as e:
                    st.error(f"K-Means failed: {e}")

        if use_gmm:
            if comms is None:
                st.error("Gaussian Mixture Model requires the number of clusters to be set.")
            else:
                try:
                    gmm_labels = run_gmm(X, comms)
                    results["Gaussian Mixture"] = gmm_labels
                except Exception as e:
                    st.error(f"Gaussian Mixture failed: {e}")

        if not results:
            st.warning("No clustering result was produced.")
        else:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)

            st.subheader("PCA Visualization")
            fig = plot_pca_results(X_pca, results)
            st.pyplot(fig)

            st.subheader("Cluster Assignments")

            for method_name, labels in results.items():
                st.markdown(f"### {method_name}")

                df_labels = pd.DataFrame({
                    "Index": np.arange(len(labels)),
                    "Cluster": labels
                })
                st.dataframe(df_labels)

                for cl in np.unique(labels):
                    inds = np.where(labels == cl)[0]
                    with st.expander(f"{method_name} - Cluster {cl}"):
                        st.write(", ".join(map(str, inds)))
