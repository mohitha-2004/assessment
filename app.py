import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors

# Simulated cross-validation data (replace with actual results)
folds = [1, 2, 3, 4, 5]
accuracies = [0.82, 0.85, 0.88, 0.83, 0.87]
precisions = [0.80, 0.82, 0.86, 0.81, 0.85]
recalls = [0.79, 0.84, 0.87, 0.82, 0.86]

df = pd.DataFrame({
    "Fold": folds,
    "Accuracy": accuracies,
    "Precision": precisions,
    "Recall": recalls
})

# Streamlit UI
st.title("🧬 AI-Powered Drug Discovery Dashboard")
st.markdown("### 📊 Cross-Validation Metrics")

# Line chart
st.line_chart(df.set_index("Fold"))

# Bar chart
st.markdown("#### 🔍 Fold-wise Accuracy (Bar Chart)")
st.bar_chart(df.set_index("Fold")["Accuracy"])

# Box plot
st.markdown("#### 📦 Box Plot of All Metrics")
fig, ax = plt.subplots()
ax.boxplot([accuracies, precisions, recalls], labels=["Accuracy", "Precision", "Recall"])
ax.set_ylabel("Scores")
st.pyplot(fig)

# Molecule Visualization
st.markdown("### 🧪 Molecule Visualization from SMILES")
smiles = st.text_input("Enter SMILES:", "CCO")  # Default: Ethanol

try:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        st.image(Draw.MolToImage(mol), caption="2D Molecule Structure")
        
        # Show basic molecular info
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
        st.warning("Invalid SMILES format. Please try again.")
except Exception as e:
    st.error(f"Error: {str(e)}")
