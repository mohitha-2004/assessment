import streamlit as st
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw

# Sample accuracy list (Replace with real values from your model)
accuracies = [0.82, 0.85, 0.88, 0.83, 0.87]

# Streamlit UI
st.title("🧬 AI-Powered Drug Discovery Dashboard")

st.markdown("### 🔁 Cross-Validation Accuracies")
st.line_chart(accuracies)

st.markdown("### 🧪 Molecule Visualization from SMILES")
smiles = st.text_input("Enter SMILES:", "CCO")  # Default = Ethanol

try:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        st.image(Draw.MolToImage(mol), caption="Molecule Structure")
    else:
        st.warning("Invalid SMILES format. Please try again.")
except Exception as e:
    st.error(f"Error: {str(e)}")
