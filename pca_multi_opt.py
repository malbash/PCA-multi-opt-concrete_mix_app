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

# Choose a professional style for plots
plt.style.use("tableau-colorblind10")  # or "seaborn", "ggplot", etc.

# -------------------------------------------------------------------
# 1. INGREDIENT ORDER
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

# -------------------------------------------------------------------
# 2. TRAIN THE MODEL (cached)
# -------------------------------------------------------------------
@st.cache_data
def load_and_train_model():
    df = pd.read_excel('cleaned.xlsx')
    df = df.dropna()

    X = df[ingredient_keys]
    y = df["F'c (MPa)"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, (mse, mae, r2)

model, (mse, mae, r2) = load_and_train_model()

# For logging or display
print(f"\nRF Model Performance: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

# -------------------------------------------------------------------
# 3. COEFFICIENTS & NAMES
# -------------------------------------------------------------------
co2_coefficients = {
    "Cement (Kg/m3)": 0.795,
    "Blast Furnace Slag (Kg/m3)": 0.135,
    "Silica Fume (Kg/m3)": 0.024,
    "Fly Ash (Kg/m3)": 0.0235,
    "Water (Kg/m3)": 0.00025,
    "Coarse Aggregate (Kg/m3)": 0.026,
    "Fine Aggregate (Kg/m3)": 0.01545,
}

cost_coefficients = {
    "Cement (Kg/m3)": 0.10,
    "Blast Furnace Slag (Kg/m3)": 0.05,
    "Silica Fume (Kg/m3)": 0.40,
    "Fly Ash (Kg/m3)": 0.03,
    "Water (Kg/m3)": 0.0005,
    "Coarse Aggregate (Kg/m3)": 0.02,
    "Fine Aggregate (Kg/m3)": 0.015,
}

cleaned_names = {
    "Cement (Kg/m3)": "Cement",
    "Blast Furnace Slag (Kg/m3)": "Blast Furnace Slag",
    "Silica Fume (Kg/m3)": "Silica Fume",
    "Fly Ash (Kg/m3)": "Fly Ash",
    "Water (Kg/m3)": "Water",
    "Coarse Aggregate (Kg/m3)": "Coarse Aggregate",
    "Fine Aggregate (Kg/m3)": "Fine Aggregate",
}

strength_ranges = {
    "Normal Strength": "NSC",
    "High Strength": "HSC",
    "Ultra-High Performance": "UHPC"
}

# Unit conversions
KG_TO_LB = 2.20462
MPA_TO_PSI = 145.038
KG_PER_M3_TO_LB_PER_FT3 = 2.20462 / 35.3147
M3_TO_FT3 = 35.3147

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
        raise ValueError(f"Unknown type: {concrete_type}")

# -------------------------------------------------------------------
# 5. Problem (with Constraint)
# -------------------------------------------------------------------
class ConcreteMixOptimizationProblem(Problem):
    """
    Minimize [CO2, Cost, -Strength], subject to Strength >= 0.9*target_strength.
    => G = 0.9*target_strength - Strength <= 0
    """
    def __init__(self, concrete_type, target_strength):
        n_vars = len(ingredient_keys)
        n_objs = 3
        n_constraints = 1

        xl = [b[0] for b in get_bounds(concrete_type)]
        xu = [b[1] for b in get_bounds(concrete_type)]

        super().__init__(n_var=n_vars,
                         n_obj=n_objs,
                         n_constr=n_constraints,
                         xl=xl,
                         xu=xu)

        self.target_strength = target_strength

    def _evaluate(self, X, out, *args, **kwargs):
        df_input = pd.DataFrame(X, columns=ingredient_keys)
        strength = model.predict(df_input)

        co2_vals = [co2_coefficients[k] for k in ingredient_keys]
        co2 = np.dot(X, co2_vals)

        cost_vals = [cost_coefficients[k] for k in ingredient_keys]
        cost = np.dot(X, cost_vals)

        f1 = co2
        f2 = cost
        f3 = -strength  # maximizing strength => minimize -strength

        out["F"] = np.column_stack([f1, f2, f3])

        # Constraint
        G = 0.9*self.target_strength - strength
        out["G"] = np.column_stack([G])

# -------------------------------------------------------------------
# 6. ALGORITHMS & RUN
# -------------------------------------------------------------------
def run_multiobjective_optimization(target_strength, concrete_type, 
                                    algorithm_name="NSGA2",
                                    pop_size=150, n_gen=100):
    problem = ConcreteMixOptimizationProblem(concrete_type, target_strength)
    termination = get_termination("n_gen", n_gen)

    if algorithm_name == "NSGA2":
        algorithm = NSGA2(pop_size=pop_size)
    elif algorithm_name == "NSGA3":
        ref_dirs = get_reference_directions("das-dennis", 3, n_points=91)
        algorithm = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")

    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=42,
                   save_history=True,
                   verbose=False)

    return res

def extract_best_gen_data(history):
    """
    For each generation, record the best (lowest) CO2,
    best (lowest) Cost, and best (highest) Strength.
    """
    best_co2, best_cost, best_strength = [], [], []

    for algo_gen in history:
        pop = algo_gen.pop
        F = pop.get("F")
        co2_min = np.min(F[:,0])
        cost_min = np.min(F[:,1])
        neg_strength_min = np.min(F[:,2])
        strength_max = -neg_strength_min

        best_co2.append(co2_min)
        best_cost.append(cost_min)
        best_strength.append(strength_max)

    return best_co2, best_cost, best_strength

# -------------------------------------------------------------------
# 7. HELPER FUNCS
# -------------------------------------------------------------------
def predict_strength(mix_metric):
    df_input = pd.DataFrame([mix_metric], columns=ingredient_keys)
    return model.predict(df_input)[0]

def select_best_solution_from_pareto(pareto_front, pareto_solutions, priority_str):
    """
    Lexicographic selection:
      f0=CO2, f1=Cost, f2=-Strength
      E.g. "CO₂ → Strength → Cost"
    """
    obj_map = {
        "CO₂": 0,
        "Cost": 1,
        "Strength": 2
    }
    parts = [p.strip() for p in priority_str.split("→")]
    parts = [p.replace(" ", "") for p in parts]
    priority_indices = [obj_map[p] for p in parts]

    pf = pareto_front.copy()
    ps = pareto_solutions.copy()

    for idx in reversed(priority_indices):
        sidx = np.argsort(pf[:, idx])
        pf = pf[sidx]
        ps = ps[sidx]

    return pf[0], ps[0]

def plot_evolution(history, algorithm_name):
    """
    Return a Matplotlib Figure showing best CO2, Cost, Strength vs generation.
    """
    best_co2_list, best_cost_list, best_strength_list = extract_best_gen_data(history)
    generations = np.arange(1, len(best_co2_list)+1)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(generations, best_co2_list, label="Best CO₂", color="red")
    ax.plot(generations, best_cost_list, label="Best Cost", color="blue")
    ax.plot(generations, best_strength_list, label="Best Strength", color="green")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Objective Value")
    ax.set_title(f"Objective Improvement: {algorithm_name}")
    ax.legend()
    ax.grid(True)

    return fig

# -------------------------------------------------------------------
# 8. STREAMLIT APP
# -------------------------------------------------------------------
def main():
    st.title("PCA - Concrete Mix Optimizer")
    st.markdown("""
    **Multi-Objective Optimization** with constraints:
    - Minimize CO₂,
    - Minimize Cost,
    - Maximize Strength (≥ 90% of target).
    """)

#    with st.expander("Model Performance"):
#        st.write(f"**Random Forest** on test set:")
#        st.write(f"- MSE: {mse:.4f}")
#        st.write(f"- MAE: {mae:.4f}")
#        st.write(f"- R²: {r2:.4f}")

    # 1) Input Type
    input_type = st.radio(
        "Select Input Type", 
        ["Target Strength", "Ingredients"], 
        index=0
    )

    # 2) If "Target Strength", we show a numeric input
    #    else we show 7 numeric inputs for ingredients
    user_strength = None
    user_mix = None
    if input_type == "Target Strength":
        user_strength = st.number_input("Target Strength (MPa)", value=30.0, min_value=0.0)
    else:
        st.write("Enter Ingredient Amounts:")
        user_mix = []
        for k in ingredient_keys:
            amt = st.number_input(f"{k}", value=0.0, min_value=0.0)
            user_mix.append(amt)

    # 3) Concrete Type
    concrete_type_display = st.selectbox(
        "Concrete Type",
        list(strength_ranges.keys()),
        index=0
    )
    # Map to NSC, HSC, UHPC
    concrete_type = strength_ranges[concrete_type_display]

    # 4) Priority
    priority_options = [
        "CO₂ → Strength → Cost",
        "Strength → CO₂ → Cost",
        "Strength → Cost → CO₂",
        "CO₂ → Cost → Strength",
        "Cost → Strength → CO₂",
        "Cost → CO₂ → Strength"
    ]
    chosen_priority = st.selectbox("Set Optimization Priority", priority_options, index=0)

    # 5) Algorithm
    algo_options = ["NSGA2", "NSGA3", "Compare All"]
    chosen_algorithm = st.selectbox("Select Algorithm", algo_options, index=0)

    # 6) Unit toggle
    units = st.radio("Units", ["Metric", "US"], index=0)

    # 7) "Run Optimization" button
    if st.button("Optimize"):
        try:
            # figure out target_strength
            if input_type == "Target Strength":
                target_strength = float(user_strength)
                if units == "US":
                    target_strength = target_strength / MPA_TO_PSI
            else:
                # user typed their own mix
                typed_mix = user_mix
                if units == "US":
                    # Convert from lb/ft³ -> Kg/m³
                    typed_mix = [v / KG_PER_M3_TO_LB_PER_FT3 for v in typed_mix]

                # Predict Strength in MPa
                predicted_strength = predict_strength(typed_mix)
                target_strength = predicted_strength
            
            # Run the chosen algorithms
            if chosen_algorithm == "Compare All":
                algos_to_run = ["NSGA2", "NSGA3"]
            else:
                algos_to_run = [chosen_algorithm]

            all_res = {}
            for alg in algos_to_run:
                # multiobjective optimization
                res = run_multiobjective_optimization(
                    target_strength=target_strength,
                    concrete_type=concrete_type,
                    algorithm_name=alg,
                    pop_size=150,
                    n_gen=100
                )
                all_res[alg] = res
            
            # For each algorithm, get best solution
            best_data = {}
            for alg in algos_to_run:
                pf = all_res[alg].F
                px = all_res[alg].X
                best_pf, best_ps = select_best_solution_from_pareto(pf, px, chosen_priority)
                co2_val = best_pf[0]
                cost_val = best_pf[1]
                strength_val = -best_pf[2]

                # Check constraint
                if strength_val < 0.9 * target_strength:
                    st.warning(f"{alg}: best solution < 90% of target strength. (Strength={strength_val:.2f}, required≥{0.9*target_strength:.2f})")

                best_data[alg] = {
                    "CO₂ Emissions": co2_val,
                    "Predicted Strength": strength_val,
                    "Mix Proportions": best_ps,
                    "Cost": cost_val
                }
            
            # 8) Display results
            if len(algos_to_run) == 1:
                # Single Algorithm
                single_alg = algos_to_run[0]
                st.subheader(f"Results: {single_alg}")
                show_results_for_algorithm(single_alg, all_res[single_alg], best_data[single_alg], units)
            else:
                # Compare All => side by side columns
                col1, col2 = st.columns(2)
                algA, algB = algos_to_run[0], algos_to_run[1]

                with col1:
                    st.subheader(f"Results: {algA}")
                    show_results_for_algorithm(algA, all_res[algA], best_data[algA], units)
                
                with col2:
                    st.subheader(f"Results: {algB}")
                    show_results_for_algorithm(algB, all_res[algB], best_data[algB], units)

        except Exception as ex:
            st.error(f"Error: {ex}")

def show_results_for_algorithm(algorithm_name, res, best_solution_data, units):
    """
    Display a text summary of the best solution,
    plus a figure of best CO2/Cost/Strength evolution.
    """
    co2_val = best_solution_data["CO₂ Emissions"]
    cost_val = best_solution_data["Cost"]
    strength_val = best_solution_data["Predicted Strength"]
    mix_vals = best_solution_data["Mix Proportions"]

    # Convert units if needed
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
        co2_disp = co2_val * KG_TO_LB / M3_TO_FT3  # lb/ft³
        strength_disp = strength_val * MPA_TO_PSI  # psi
        cost_disp = cost_val / M3_TO_FT3  # $/ft³
        mix_converted = [v * KG_PER_M3_TO_LB_PER_FT3 for v in mix_vals]
        co2_unit = "lb"
        strength_unit = "Psi"
        cost_volume_unit = "per ft³"
        mix_unit = "lb/ft³"

    # Text summary
    st.write(f"**Best Solution** (by chosen priority):")
    st.write(f"- CO₂: {co2_disp:.2f} {co2_unit}")
    st.write(f"- Strength: {strength_disp:.2f} {strength_unit}")
    st.write(f"- Cost: ${cost_disp:.2f} {cost_volume_unit}")
    st.write("**Mix Proportions**:")
    for ingr_name, val_m in zip(ingredient_keys, mix_converted):
        st.write(f"• {cleaned_names[ingr_name]} ({mix_unit}): {val_m:.2f}")

    # Plot evolution
    fig = plot_evolution(res.history, algorithm_name)
    st.pyplot(fig)

# -------------------------------------------------------------------
# RUN THE APP
# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
