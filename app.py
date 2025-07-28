# Import libraries
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import zipfile
import requests
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors

# ----------------------------------------
# Download and extract MUTAG dataset
# ----------------------------------------
@st.cache_data
def download_and_extract_mutag():
    url = "https://www.chrsmrrs.com/graphkerneldatasets/MUTAG.zip"
    if not os.path.exists("MUTAG.zip"):
        st.info("Downloading MUTAG.zip...")
        r = requests.get(url)
        with open("MUTAG.zip", "wb") as f:
            f.write(r.content)

    with zipfile.ZipFile("MUTAG.zip", "r") as zip_ref:
        zip_ref.extractall("mutag_data")

# ----------------------------------------
# Load features and labels from extracted files
# ----------------------------------------
@st.cache_data
def load_mutag_features_labels():
    base_path = "mutag_data/MUTAG/"
    indicator_path = base_path + "MUTAG_graph_indicator.txt"
    labels_path = base_path + "MUTAG_graph_labels.txt"
    node_labels_path = base_path + "MUTAG_node_labels.txt"


    if not all(os.path.exists(p) for p in [indicator_path, labels_path, node_labels_path]):
        raise FileNotFoundError("Required MUTAG files not found after extraction.")

    with open(indicator_path) as f:
        graph_indicator = [int(x.strip()) for x in f]

    with open(labels_path) as f:
        graph_labels = [int(x.strip()) for x in f]

    with open(node_labels_path) as f:
        node_labels = [int(x.strip()) for x in f]

    graphs = {}
    for node_idx, graph_id in enumerate(graph_indicator):
        if graph_id not in graphs:
            graphs[graph_id] = []
        graphs[graph_id].append(node_labels[node_idx])

    # Use histogram of node labels as feature vector for each graph
    features = []
    for graph_nodes in graphs.values():
        hist = np.bincount(graph_nodes, minlength=7)  # assuming node labels <= 6
        features.append(hist)

    X = np.array(features)
    y = np.array(graph_labels)
    return X, y

# ----------------------------------------
# Title
# ----------------------------------------
st.title("🧬 AI-Powered Drug Discovery Dashboard")

# ----------------------------------------
# Load and analyze MUTAG dataset
# ----------------------------------------
if st.button("📥 Load and Analyze MUTAG Dataset"):
    with st.spinner("Downloading and analyzing MUTAG dataset..."):
        download_and_extract_mutag()
        X, y = load_mutag_features_labels()

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        accs, precs, recs = [], [], []
        for train_idx, test_idx in kf.split(X):
            model = RandomForestClassifier()
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[test_idx])
            accs.append(accuracy_score(y[test_idx], y_pred))
            precs.append(precision_score(y[test_idx], y_pred, zero_division=0))
            recs.append(recall_score(y[test_idx], y_pred, zero_division=0))

        folds = [1, 2, 3, 4, 5]
        df = pd.DataFrame({
            "Fold": folds,
            "Accuracy": accs,
            "Precision": precs,
            "Recall": recs
        })

        st.success("✅ Cross-validation complete!")

        # Display metrics
        st.markdown("### 📊 Cross-Validation Metrics")
        st.line_chart(df.set_index("Fold"))

        st.markdown("#### 🔍 Fold-wise Accuracy (Bar Chart)")
        st.bar_chart(df.set_index("Fold")["Accuracy"])

        st.markdown("#### 📦 Box Plot of All Metrics")
        fig, ax = plt.subplots()
        ax.boxplot([accs, precs, recs], tick_labels=["Accuracy", "Precision", "Recall"])

        ax.set_ylabel("Scores")
        st.pyplot(fig)

# ----------------------------------------
# Molecule Visualization from SMILES
# ----------------------------------------
st.markdown("### 🧪 Molecule Visualization from SMILES")
smiles = st.text_input("Enter SMILES:", "CCO")  # Default: Ethanol

try:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        st.image(Draw.MolToImage(mol), caption="2D Molecule Structure")

        # Molecular Properties
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)

        st.markdown("### 📐 Molecular Properties")
        st.write(f"**Molecular Weight:** {mw:.2f} g/mol")
        st.write(f"**LogP (Octanol-Water Partition):** {logp:.2f}")
        st.write(f"**H-Bond Donors:** {hbd}")
        st.write(f"**H-Bond Acceptors:** {hba}")
    else:
        st.warning("❗ Invalid SMILES format. Please try again.")
except Exception as e:
    st.error(f"An error occurred while processing the molecule: {e}")
