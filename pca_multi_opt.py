import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

# -------------------------------------------------------------------
# 1. Widen the number_input boxes via CSS
# -------------------------------------------------------------------
st.markdown("""
    <style>
    /* Make number_input fields wider */
    div[data-testid="stNumberInput"] > label > div > input {
        width: 200px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Choose a professional style for plots
plt.style.use("tableau-colorblind10")

# -------------------------------------------------------------------
# 2. Define INGREDIENT ORDER + DICTS
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

# These dictionaries will store user-updated coefficients
co2_dict = default_co2_coeff.copy()
cost_dict = default_cost_coeff.copy()

strength_ranges = {
    "Normal Strength": "NSC",
    "High Strength": "HSC",
    "Ultra-High Performance": "UHPC"
}

# Conversion constants
KG_TO_LB = 2.20462
MPA_TO_PSI = 145.038
KG_PER_M3_TO_LB_PER_FT3 = 2.20462 / 35.3147
M3_TO_FT3 = 35.3147

# -------------------------------------------------------------------
# 3. LOAD DATA & TRAIN MODEL (CACHED)
# -------------------------------------------------------------------
@st.cache_data
def load_and_train_model():
    df = pd.read_excel('cleaned.xlsx')  # Must exist in same folder
    df = df.dropna()

    X = df[ingredient_keys]
    y = df["F'c (MPa)"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred_optimized = model.predict(X_test)
    mse_val = mean_squared_error(y_test, y_pred_optimized)
    mae_val = mean_absolute_error(y_test, y_pred_optimized)
    r2_val = r2_score(y_test, y_pred_optimized)

    return model, (mse_val, mae_val, r2_val)

model, (mse, mae, r2) = load_and_train_model()

# -------------------------------------------------------------------
# 4. BOUNDS
# -------------------------------------------------------------------
def get_bounds(concrete_type):
    if concrete_type == "NSC":
        return [
            (140, 400),
            (0, 150),
            (0, 1),
            (0, 100),
            (120, 200),
            (800, 1200),
            (600, 700)
        ]
    elif concrete_type == "HSC":
        return [
            (240, 550),
            (0, 150),
            (0, 50),
            (0, 150),
            (105, 160),
            (700, 1000),
            (600, 800)
        ]
    elif concrete_type == "UHPC":
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
        st.error(f"Unknown type: {concrete_type}")
        st.stop()

# -------------------------------------------------------------------
# 5. PROBLEM CLASS
# -------------------------------------------------------------------
class ConcreteMixOptimizationProblem(Problem):
    def __init__(self, concrete_type, target_strength, co2_dict, cost_dict):
        n_vars = len(ingredient_keys)
        n_objs = 3
        n_constr = 1

        bounds = get_bounds(concrete_type)
        xl = [b[0] for b in bounds]
        xu = [b[1] for b in bounds]

        super().__init__(
            n_var=n_vars,
            n_obj=n_objs,
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
        neg_strength = -strength_vals

        out["F"] = np.column_stack([co2, cost, neg_strength])
        # Strength constraint => 0.9 * target_strength - strength <= 0
        G = 0.9 * self.target_strength - strength_vals
        out["G"] = np.column_stack([G])

# -------------------------------------------------------------------
# 6. ALGORITHMS
# -------------------------------------------------------------------
def run_multiobjective_optimization(target_strength, concrete_type, 
                                    co2_dict, cost_dict,
                                    algorithm_name="NSGA2",
                                    pop_size=150, n_gen=100):
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.termination import get_termination
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.optimize import minimize

    problem = ConcreteMixOptimizationProblem(
        concrete_type, target_strength, co2_dict, cost_dict
    )
    termination = get_termination("n_gen", n_gen)

    if algorithm_name == "NSGA2":
        algorithm = NSGA2(pop_size=pop_size)
    elif algorithm_name == "NSGA3":
        ref_dirs = get_reference_directions("das-dennis", 3, n_points=91)
        algorithm = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)
    else:
        st.error(f"Unsupported algorithm: {algorithm_name}")
        st.stop()

    res = minimize(
        problem, algorithm, termination, seed=42, verbose=False, save_history=True
    )
    return res

def extract_best_gen_data(history):
    best_co2_list, best_cost_list, best_strength_list = [], [], []
    for gen_data in history:
        pop = gen_data.pop
        F = pop.get("F")  # shape: (pop_size, 3)
        co2_min = np.min(F[:,0])
        cost_min = np.min(F[:,1])
        neg_strength_min = np.min(F[:,2])
        strength_max = -neg_strength_min
        best_co2_list.append(co2_min)
        best_cost_list.append(cost_min)
        best_strength_list.append(strength_max)
    return best_co2_list, best_cost_list, best_strength_list

def select_best_solution_from_pareto(pareto_front, pareto_solutions, priority_str):
    """
    Lexicographic selection from the front: f0=CO2, f1=Cost, f2=-Strength
    E.g. "CO₂ → Strength → Cost"
    """
    obj_map = {"CO₂":0, "Cost":1, "Strength":2}
    parts = [p.strip().replace(" ","") for p in priority_str.split("→")]
    indices = [obj_map[x] for x in parts]

    pf = pareto_front.copy()
    ps = pareto_solutions.copy()

    # stable sort from last priority to first
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
        mix_converted = [x * KG_PER_M3_TO_LB_PER_FT3 for x in mix_vals]
        co2_unit = "lb"
        strength_unit = "Psi"
        cost_volume_unit = "per ft³"
        mix_unit = "lb/ft³"

    st.write(f"**Algorithm: {algorithm_name}**")
    st.write(f"• CO₂ Emissions: {co2_disp:.3f} {co2_unit}")
    st.write(f"• Strength: {strength_disp:.3f} {strength_unit}")
    st.write(f"• Cost: ${cost_disp:.3f} {cost_volume_unit}")
    st.write("**Mix Proportions:**")
    for ingr, val_m in zip(ingredient_keys, mix_converted):
        nm = cleaned_names[ingr]
        st.write(f"- {nm} ({mix_unit}): {val_m:.2f}")

    fig = plot_evolution(res.history, algorithm_name)
    st.pyplot(fig)

# -------------------------------------------------------------------
# MAIN STREAMLIT APP
# -------------------------------------------------------------------
def main():
    st.title("PCA - Concrete Mix Optimizer")

    # Show model performance in an expander
    with st.expander("Model Performance"):
        st.write(f"RF Performance: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

    # Input Type
    input_type = st.radio("Select Input Type:", ["Target Strength", "Ingredients"], index=0)

    # If Target Strength, show one numeric input
    # If Ingredients, show numeric inputs for all 7 ingredients
    user_strength = None
    typed_mix = None
    if input_type == "Target Strength":
        user_strength = st.number_input("Target Strength (MPa)", value=30.0, min_value=0.0)
    else:
        st.write("Enter Ingredient Amounts (Kg/m³):")
        typed_mix = []
        for ingr_key in ingredient_keys:
            val = st.number_input(f"{cleaned_names[ingr_key]}", value=0.0, min_value=0.0)
            typed_mix.append(val)

    # Unit toggle
    units = st.radio("Units:", ["Metric", "US"], index=0)

    # Concrete Type
    type_options = list(strength_ranges.keys())
    chosen_type = st.selectbox("Concrete Type:", type_options, index=0)
    concrete_type_key = strength_ranges[chosen_type]

    # Priority
    priority_choices = [
        "CO₂ → Strength → Cost",
        "Strength → CO₂ → Cost",
        "Strength → Cost → CO₂",
        "CO₂ → Cost → Strength",
        "Cost → Strength → CO₂",
        "Cost → CO₂ → Strength"
    ]
    chosen_priority = st.selectbox("Optimization Priority", priority_choices, index=0)

    # Algorithm
    algo_list = ["NSGA2", "NSGA3", "Compare All"]
    chosen_alg = st.selectbox("Select Algorithm", algo_list, index=0)

    # ADVANCED SETTINGS (collapsible) at the bottom
    st.write("---")
    with st.expander("Advanced Settings (Optional)", expanded=False):
        st.write("Override default CO₂ and Cost coefficients if needed:")

        # We'll do a small "table" style
        cA, cB, cC = st.columns([2,1.5,1.5])
        cA.write("**Ingredient**")
        cB.write("**CO₂ Coeff (Kg-CO₂/Kg)**")
        cC.write("**Cost Coeff (USD/Kg)**")

        for ingr_key in ingredient_keys:
            col1, col2, col3 = st.columns([2,1.5,1.5])
            col1.write(f"{cleaned_names[ingr_key]}")
            new_co2 = col2.number_input(
                label="",
                value=float(co2_dict[ingr_key]),
                step=0.0001,
                format="%.6f",
                key=f"co2_{ingr_key}"
            )
            new_cost = col3.number_input(
                label="",
                value=float(cost_dict[ingr_key]),
                step=0.0001,
                format="%.6f",
                key=f"cost_{ingr_key}"
            )
            co2_dict[ingr_key] = new_co2
            cost_dict[ingr_key] = new_cost

    if st.button("Optimize"):
        try:
            # Figure out final target_strength
            if input_type == "Target Strength":
                # Convert if US
                final_strength = float(user_strength)
                if units == "US":
                    final_strength = final_strength / MPA_TO_PSI
            else:
                # typed mix
                if units == "US":
                    typed_mix = [v / KG_PER_M3_TO_LB_PER_FT3 for v in typed_mix]
                # Predict
                final_strength = model.predict(pd.DataFrame([typed_mix], columns=ingredient_keys))[0]

            # Decide if Compare All
            if chosen_alg == "Compare All":
                to_run = ["NSGA2", "NSGA3"]
            else:
                to_run = [chosen_alg]

            results = {}
            best_data = {}
            for alg in to_run:
                res = run_multiobjective_optimization(
                    target_strength=final_strength,
                    concrete_type=concrete_type_key,
                    co2_dict=co2_dict,
                    cost_dict=cost_dict,
                    algorithm_name=alg,
                    pop_size=150,
                    n_gen=100
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
