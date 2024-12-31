import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

plt.style.use("tableau-colorblind10")

# CSS hack to widen number_input boxes
st.markdown("""
    <style>
    div[data-testid="stNumberInput"] > label > div > input {
        width: 200px !important;
    }
    </style>
""", unsafe_allow_html=True)

# -- Sample Ingredient Setup --
ingredient_keys = [
    "Cement (Kg/m3)",
    "Blast Furnace Slag (Kg/m3)",
    "Silica Fume (Kg/m3)",
    "Fly Ash (Kg/m3)",
    "Water (Kg/m3)",
    "Coarse Aggregate (Kg/m3)",
    "Fine Aggregate (Kg/m3)"
]

cleaned_names = {
    "Cement (Kg/m3)": "Cement",
    "Blast Furnace Slag (Kg/m3)": "Blast Furnace Slag",
    "Silica Fume (Kg/m3)": "Silica Fume",
    "Fly Ash (Kg/m3)": "Fly Ash",
    "Water (Kg/m3)": "Water",
    "Coarse Aggregate (Kg/m3)": "Coarse Aggregate",
    "Fine Aggregate (Kg/m3)": "Fine Aggregate",
}

default_co2_coeff = {
    "Cement (Kg/m3)": 0.795,
    "Blast Furnace Slag (Kg/m3)": 0.135,
    "Silica Fume (Kg/m3)": 0.024,
    "Fly Ash (Kg/m3)": 0.0235,
    "Water (Kg/m3)": 0.00025,
    "Coarse Aggregate (Kg/m3)": 0.026,
    "Fine Aggregate (Kg/m3)": 0.01545,
}
default_cost_coeff = {
    "Cement (Kg/m3)": 0.10,
    "Blast Furnace Slag (Kg/m3)": 0.05,
    "Silica Fume (Kg/m3)": 0.40,
    "Fly Ash (Kg/m3)": 0.03,
    "Water (Kg/m3)": 0.0005,
    "Coarse Aggregate (Kg/m3)": 0.02,
    "Fine Aggregate (Kg/m3)": 0.015,
}

co2_dict = default_co2_coeff.copy()
cost_dict = default_cost_coeff.copy()

# ... imagine your model load, etc. is here ...
def load_and_train_model():
    # Just a placeholder
    return None, (0.0, 0.0, 1.0)

model, (mse, mae, r2) = load_and_train_model()

def main():
    st.title("Concrete Mix Optimizer")

    # A. Input Type
    input_type = st.radio("Select Input Type:", ["Target Strength", "Ingredients"], index=0)
    if input_type == "Target Strength":
        user_strength = st.number_input("Target Strength (MPa)", value=30.0, min_value=0.0)
    else:
        st.write("Enter Ingredient Amounts (Kg/m³):")
        # Your code for typed mix

    # B. Concrete Type, Priority, Algorithm, etc.
    st.write("... your other UI elements ...")

    # =========================
    #  ADVANCED SETTINGS  (move near bottom)
    # =========================
    st.subheader("Advanced Settings")
    st.markdown("You can override default CO₂ and Cost coefficients here.")

    # We use an open 'expander' or simply place them in a container
    # We'll do a small heading row
    # let's replicate a "table" style approach:

    # Header row with 3 columns: Ingredient, CO₂, Cost
    c1, c2, c3 = st.columns([2, 1.5, 1.5])
    c1.markdown("**Ingredient**")
    c2.markdown("**CO₂ Coeff (Kg-CO₂/Kg)**")
    c3.markdown("**Cost Coeff (USD/Kg)**")

    for ingr_key in ingredient_keys:
        col1, col2, col3 = st.columns([2, 1.5, 1.5])
        col1.write(f"{cleaned_names[ingr_key]}")
        co2_val = col2.number_input(
            label="",
            value=float(co2_dict[ingr_key]),
            key=f"co2_{ingr_key}",
            step=0.0001,
            format="%.6f"
        )
        cost_val = col3.number_input(
            label="",
            value=float(cost_dict[ingr_key]),
            key=f"cost_{ingr_key}",
            step=0.0001,
            format="%.6f"
        )
        co2_dict[ingr_key] = co2_val
        cost_dict[ingr_key] = cost_val

    # =======================
    # Optimize button at the bottom
    # =======================
    if st.button("Optimize"):
        st.write("Running optimization with custom coefficients...")
        st.write("CO₂:", co2_dict)
        st.write("Cost:", cost_dict)
        # ... your code to run NSGA2 / NSGA3 ...
        st.success("Optimization done!")


if __name__ == "__main__":
    main()
