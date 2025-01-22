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

# 1) Apply a style to make the input boxes wider via CSS
st.markdown("""
    <style>
    /* Increase width for ALL number_input boxes */
    div[data-testid="stNumberInput"] > label > div > input {
        width: 250px !important;
    }
    </style>
""", unsafe_allow_html=True)

# 2) Use a consistent style for plots
plt.style.use("tableau-colorblind10")

# -------------------------------------------------------------------
# A. INGREDIENT & MODEL SETUP
# -------------------------------------------------------------------
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

# Default coefficients
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

# Keep user-updated values in global dictionaries
co2_dict = default_co2_coeff.copy()
cost_dict = default_cost_coeff.copy()

# For unit conversions
KG_TO_LB = 2.20462
MPA_TO_PSI = 145.038
KG_PER_M3_TO_LB_PER_FT3 = 2.20462 / 35.3147
M3_TO_FT3 = 35.3147

strength_ranges = {
    "Normal Strength": "NSC",
    "High Strength": "HSC",
    "Ultra-High Performance": "UHPC"
}

@st.cache_data
def load_and_train_model():
    df = pd.read_excel("cleaned.xlsx")  # must exist in same folder
    df = df.dropna()

    X = df[ingredient_keys]
    y = df["F'c (MPa)"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse_val = mean_squared_error(y_test, y_pred)
    mae_val = mean_absolute_error(y_test, y_pred)
    r2_val = r2_score(y_test, y_pred)

    return model, (mse_val, mae_val, r2_val)

model, (mse, mae, r2) = load_and_train_model()

# -------------------------------------------------------------------
# B. DEFINE Multi-Objective PROBLEM
# -------------------------------------------------------------------
def get_bounds(concrete_type_key):
    if concrete_type_key == "NSC":
        return [
            (140, 400),
            (0, 150),
            (0, 1),
            (0, 100),
            (120, 200),
            (800, 1200),
            (600, 700)
        ]
    elif concrete_type_key == "HSC":
        return [
            (240, 550),
            (0, 150),
            (0, 50),
            (0, 150),
            (105, 160),
            (700, 1000),
            (600, 800)
        ]
    elif concrete_type_key == "UHPC":
        return [
            (350, 1000),
            (0, 150),
            (140, 300),
            (0, 200),
            (125, 150),
            (0, 1),
            (650, 1200)
        ]
    else:
        st.error(f"Unknown type: {concrete_type_key}")
        st.stop()

class ConcreteMixOptimizationProblem(Problem):
    """
    Minimize [CO2, Cost, -Strength], with constraint Strength >= 0.9*target_strength
    => G = 0.9*target_strength - Strength <= 0
    """
    def __init__(self, concrete_type_key, target_strength, co2_dict, cost_dict):
        n_vars = len(ingredient_keys)
        n_obj = 3
        n_constr = 1

        bounds = get_bounds(concrete_type_key)
        xl = [b[0] for b in bounds]
        xu = [b[1] for b in bounds]

        super().__init__(
            n_var=n_vars,
            n_obj=n_obj,
            n_constr=n_constr,
            xl=xl,
            xu=xu
        )

        self.target_strength = target_strength
        self.co2_dict = co2_dict
        self.cost_dict = cost_dict

    def _evaluate(self, X, out, *args, **kwargs):
        df_input = pd.DataFrame(X, columns=ingredient_keys)
        strength_vals = model.predict(df_input)

        co2_vals = [self.co2_dict[k] for k in ingredient_keys]
        cost_vals = [self.cost_dict[k] for k in ingredient_keys]

        co2 = np.dot(X, co2_vals)
        cost = np.dot(X, cost_vals)
        neg_strength = -strength_vals  # maximize => minimize negative

        out["F"] = np.column_stack([co2, cost, neg_strength])

        # Strength constraint => 0.9*target_strength - strength <= 0
        G = 0.9*self.target_strength - strength_vals
        out["G"] = np.column_stack([G])

def run_multiobjective_optimization(target_strength, concrete_type_key,
                                    co2_dict, cost_dict,
                                    algorithm="NSGA2",
                                    pop_size=60, n_gen=40):
    problem = ConcreteMixOptimizationProblem(concrete_type_key, target_strength, co2_dict, cost_dict)
    termination = get_termination("n_gen", n_gen)

    if algorithm == "NSGA2":
        algo = NSGA2(pop_size=pop_size)
    elif algorithm == "NSGA3":
        ref_dirs = get_reference_directions("das-dennis", 3, n_points=91)
        algo = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)
    else:
        st.error(f"Unsupported algorithm: {algorithm}")
        st.stop()

    res = minimize(problem, algo, termination, seed=42, verbose=False, save_history=True)
    return res

# Extraction of best solutions over generations
def extract_best_gen_data(history):
    best_co2, best_cost, best_strength = [], [], []
    for gen_data in history:
        pop = gen_data.pop
        F = pop.get("F")
        min_co2 = np.min(F[:,0])
        min_cost = np.min(F[:,1])
        max_strength = -np.min(F[:,2])  # revert sign
        best_co2.append(min_co2)
        best_cost.append(min_cost)
        best_strength.append(max_strength)
    return best_co2, best_cost, best_strength

def select_best_solution_from_pareto(pareto_front, pareto_solutions, priority_str):
    """
    Lexicographic: f0=CO2, f1=Cost, f2=-Strength
    """
    obj_map = {"CO₂": 0, "Cost": 1, "Strength": 2}
    parts = [p.strip().replace(" ","") for p in priority_str.split("→")]
    indices = [obj_map[p] for p in parts]

    pf = pareto_front.copy()
    ps = pareto_solutions.copy()

    # stable sort from last to first
    for idx in reversed(indices):
        order = np.argsort(pf[:, idx])
        pf = pf[order]
        ps = ps[order]

    return pf[0], ps[0]

def plot_evolution(history, algorithm_name):
    best_co2, best_cost, best_strength = extract_best_gen_data(history)
    gens = np.arange(1, len(best_co2)+1)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(gens, best_co2, label="Best CO₂", color="red")
    ax.plot(gens, best_cost, label="Best Cost", color="blue")
    ax.plot(gens, best_strength, label="Best Strength", color="green")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Objective Value")
    ax.set_title(f"Objective Improvement: {algorithm_name}")
    ax.legend()
    ax.grid(True)
    return fig

def show_results_for_algorithm(algorithm_name, res, best_solution, units):
    co2_val = best_solution["CO₂ Emissions"]
    cost_val = best_solution["Cost"]
    strength_val = best_solution["Predicted Strength"]
    mix_vals = best_solution["Mix Proportions"]

    if units == "Metric":
        co2_disp = co2_val
        strength_disp = strength_val
        cost_disp = cost_val
        mix_converted = mix_vals
        co2_unit = "kg"
        strength_unit = "MPa"
        cost_volume_unit = "per m³"
        mix_unit = "Kg/m³"
    else:
        co2_disp = co2_val * KG_TO_LB / M3_TO_FT3
        strength_disp = strength_val * MPA_TO_PSI
        cost_disp = cost_val / M3_TO_FT3
        mix_converted = [m * KG_PER_M3_TO_LB_PER_FT3 for m in mix_vals]
        co2_unit = "lb"
        strength_unit = "Psi"
        cost_volume_unit = "per ft³"
        mix_unit = "lb/ft³"

    st.write(f"**Algorithm: {algorithm_name}**")
    st.write(f"**CO₂ Emissions:** {co2_disp:.3f} {co2_unit}")
    st.write(f"**Strength:** {strength_disp:.3f} {strength_unit}")
    st.write(f"**Cost:** ${cost_disp:.3f} {cost_volume_unit}")
    st.write("**Mix Proportions:**")
    for ingr_name, val_m in zip(ingredient_keys, mix_converted):
        nice_name = cleaned_names[ingr_name]
        st.write(f"- {nice_name} ({mix_unit}): {val_m:.2f}")

    fig = plot_evolution(res.history, algorithm_name)
    st.pyplot(fig)

# -------------------------------------------------------------------
# MAIN STREAMLIT APP
# -------------------------------------------------------------------
def main():
    st.title("Concrete Mix Optimizer")

    # Show model performance
    with st.expander("Model Performance"):
        st.write(f"MSE: {mse:.4f}")
        st.write(f"MAE: {mae:.4f}")
        st.write(f"R²: {r2:.4f}")

    # ADVANCED SETTINGS for CO2 & Cost
    st.subheader("Advanced Settings")
    st.markdown("Adjust default coefficients if needed.")
    with st.expander("Customize CO₂ & Cost Coefficients"):
        # We'll show a table-like layout
        st.write("**Ingredient | CO₂ Coeff (Kg-CO₂/Kg) | Cost Coeff (USD/Kg)**")
        for ingr in ingredient_keys:
            # 3 columns for better spacing
            c1, c2, c3 = st.columns([2,1,1])
            with c1:
                st.write(f"**{cleaned_names[ingr]}**")
            with c2:
                val_co2 = st.number_input(
                    f"CO₂ for {ingr}",
                    value=float(co2_dict[ingr]),
                    step=0.001,
                    format="%.6f",
                    key=f"co2_{ingr}"
                )
            with c3:
                val_cost = st.number_input(
                    f"Cost for {ingr}",
                    value=float(cost_dict[ingr]),
                    step=0.001,
                    format="%.6f",
                    key=f"cost_{ingr}"
                )
            co2_dict[ingr] = val_co2
            cost_dict[ingr] = val_cost

    # Input type: target strength or manual ingredients
    input_type = st.radio("Select Input Type:", ["Target Strength", "Ingredients"], index=0)
    target_strength = None
    typed_mix = None

    if input_type == "Target Strength":
        target_strength = st.number_input("Target Strength (MPa)", value=30.0, min_value=0.0)
    else:
        st.write("Enter Ingredient Amounts (Kg/m³):")
        typed_mix = []
        for ingr in ingredient_keys:
            amt = st.number_input(f"{cleaned_names[ingr]}", value=0.0, min_value=0.0)
            typed_mix.append(amt)

    # Concrete Type
    concrete_types = list(strength_ranges.keys())
    chosen_type = st.selectbox("Concrete Type:", concrete_types, index=0)
    concrete_type_key = strength_ranges[chosen_type]

    # Priority
    priority_options = [
        "CO₂ → Strength → Cost",
        "Strength → CO₂ → Cost",
        "Strength → Cost → CO₂",
        "CO₂ → Cost → Strength",
        "Cost → Strength → CO₂",
        "Cost → CO₂ → Strength"
    ]
    chosen_priority = st.selectbox("Optimization Priority", priority_options, index=0)

    # Algorithm
    algo_options = ["NSGA2", "NSGA3", "Compare All"]
    chosen_alg = st.selectbox("Algorithm", algo_options, index=0)

    # Units
    units = st.radio("Units", ["Metric", "US"], index=0)

    # Optimize button
    if st.button("Optimize"):
        # Convert target strength if user typed mix
        try:
            if input_type == "Target Strength":
                if units == "US":
                    target_strength = float(target_strength) / MPA_TO_PSI
            else:
                # user typed mix
                if units == "US":
                    typed_mix = [val / KG_PER_M3_TO_LB_PER_FT3 for val in typed_mix]
                # Predict strength
                predicted_strength = model.predict(pd.DataFrame([typed_mix], columns=ingredient_keys))[0]
                target_strength = predicted_strength

            # Decide which algorithms to run
            if chosen_alg == "Compare All":
                to_run = ["NSGA2", "NSGA3"]
            else:
                to_run = [chosen_alg]

            results = {}
            best_data = {}
            for alg in to_run:
                res = run_multiobjective_optimization(
                    target_strength=target_strength,
                    concrete_type_key=concrete_type_key,
                    co2_dict=co2_dict,
                    cost_dict=cost_dict,
                    algorithm=alg,
                    pop_size=60,
                    n_gen=40
                )
                results[alg] = res

                pf = res.F
                px = res.X
                best_pf, best_ps = select_best_solution_from_pareto(pf, px, chosen_priority)
                co2_val = best_pf[0]
                cost_val = best_pf[1]
                strength_val = -best_pf[2]
                best_data[alg] = {
                    "CO₂ Emissions": co2_val,
                    "Cost": cost_val,
                    "Predicted Strength": strength_val,
                    "Mix Proportions": best_ps
                }

            # Display
            if len(to_run) == 1:
                st.subheader(f"Results: {to_run[0]}")
                show_results_for_algorithm(to_run[0], results[to_run[0]], best_data[to_run[0]], units)
            else:
                colA, colB = st.columns(2)
                algA, algB = to_run[0], to_run[1]

                with colA:
                    st.subheader(f"Results: {algA}")
                    show_results_for_algorithm(algA, results[algA], best_data[algA], units)
                with colB:
                    st.subheader(f"Results: {algB}")
                    show_results_for_algorithm(algB, results[algB], best_data[algB], units)

        except Exception as ex:
            st.error(f"Error: {ex}")

if __name__ == "__main__":
    main()
