from generate_data import generate_patient_csv
from generate_data import load_patients_from_csv
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from Optimization_model import create_optimization_model
from Optimization_model import visualize_schedule
from adaptive_scheduling import evaluate_offline_objective_over_iterations
from adaptive_scheduling import plot_metrics_vs_time
import matplotlib.pyplot as plt
import os

from dataclasses import dataclass

@dataclass
class IterationResults:
    x: dict
    w: dict
    s: dict
    y: dict
    obj: float

def iterative_solver_bias(csv_path, T, delta, tau, z, Binary = False):
    """
    Iterative solver that only solves the problem and records:
    - I, S, W, Obj values
    - Decision variables (x, w, s, y)
    No offline evaluation is performed here.
    """
    np.random.seed(z)
    df = pd.read_csv(csv_path)
    P = df["patient_id"].tolist()
    mu = {row["patient_id"]: row["d_p"] for _, row in df.iterrows()}
    sigma = {row["patient_id"]: int(row["nu_p"]) for _, row in df.iterrows()}
    sigma_std = {row["patient_id"]: row["sigma_p"] for _, row in df.iterrows()}

    kappa = {t: np.random.randint(6, 8) for t in T}
    beta = {(p, t): (np.sqrt(t*t*t + p) )/ (t + 2)  for p in P for t in T}
    gamma = {(p, i, l): 10 * np.sqrt((i-l+6)*(i-l+5))
             for p in P for l in range(1, max(sigma.values()) - 1) for i in range(0, l)}

    fixed_x = {(p, t): 0 for p in P for t in T}
    last_Delta = {p: int(np.clip(np.random.normal(mu[p], sigma_std[p]), 1, delta)) for p in P}

    I_vals, S_vals, W_vals, Obj_vals = [], [], [], []
    var_solutions = []  # will store (x, w, s, y) for each iteration

    # Before the loop, randomly choose which patient gets the bias and whether it's added or subtracted
    biased_patient = np.random.choice(P)
    bias_sign = np.random.choice([-1, 1], p=[0.2, 0.8])  # 80% +1, 20% -1

    for t_fixed in range(len(T) - 1):
        Delta = {}
        
        # Calculate time-decreasing bias percentage (from 50% to 10%)
        bias_percentage = 0.5 - 0.2 * (t_fixed / (len(T) - 2))  # 50% to 10%
        
        for p in P:
            # Calculate the bias for this patient's mu
            if p == biased_patient:
                bias_amount = bias_sign * mu[p] * bias_percentage
                biased_mu = mu[p] + bias_amount
            else:
                biased_mu = mu[p]
            
            if t_fixed == 0:
                Delta[p] = int(np.clip(np.random.normal(biased_mu, sigma_std[p]), 1, delta))
            elif fixed_x[p, t_fixed - 1] > 0.2:
                sigma_std[p] = max(0.7, sigma_std[p] - 1.1)
                Delta[p] = int(np.clip(np.random.normal(biased_mu, sigma_std[p]), 1, delta))
            else:
                Delta[p] = last_Delta[p]
            last_Delta[p] = Delta[p]
        
        alpha = {(p, j): 50 * (Delta[p] + 1 - j) * (Delta[p] >= j)
                for p in P for j in range(1, len(T))}
    
        print(f"Solving for t_fixed = {t_fixed}")
        model = create_optimization_model(P, T, sigma, kappa, delta, alpha, beta, gamma, fixed_x, t_fixed, tau, Binary)
        model.optimize()

        if model.status == GRB.OPTIMAL:
            I = sum(gamma.get((p, i, l), 0.0) * model.getVarByName(f"y[{p},{i},{l}]").x
                    for p in P for l in range(1, sigma[p] - 1) for i in range(0, l))
            S = sum(alpha.get((p, j), 0.0) * (1 - model.getVarByName(f"s[{p},{j}]").x)
                    for p in P for j in range(1, delta)
                    if model.getVarByName(f"s[{p},{j}]") is not None)
            W = sum(beta.get((p, t), 0.0) * (1 - model.getVarByName(f"w[{p},{t}]").x)
                    for p in P for t in range(sigma[p])
                    if model.getVarByName(f"w[{p},{t}]") is not None)

            for p in P:
                for t in range(t_fixed, min(t_fixed + tau, len(T))):
                    var = model.getVarByName(f"x[{p},{t}]")
                    if var:
                        fixed_x[p, t] = (var.x)

            I_vals.append(I)
            S_vals.append(S)
            W_vals.append(W)
            Obj_vals.append(I + S + W)

            # store decision variable solutions for offline evaluation later
            var_solutions.append({
                'x': {(p, t): model.getVarByName(f"x[{p},{t}]").x for p in P for t in T if model.getVarByName(f"x[{p},{t}]")},
                'w': {(p, t): model.getVarByName(f"w[{p},{t}]").x for p in P for t in T if model.getVarByName(f"w[{p},{t}]")},
                's': {(p, j): model.getVarByName(f"s[{p},{j}]").x for p in P for j in range(1, delta) if model.getVarByName(f"s[{p},{j}]")},
                'y': {(p, i, l): model.getVarByName(f"y[{p},{i},{l}]").x for p in P for l in range(1, sigma[p] - 1) for i in range(0, l)
                      if model.getVarByName(f"y[{p},{i},{l}]")}
            })
        else:
            print(f"Infeasible at t_fixed = {t_fixed}")
            I_vals.append(None)
            S_vals.append(None)
            W_vals.append(None)
            Obj_vals.append(None)
            var_solutions.append({'x': {}, 'w': {}, 's': {}, 'y': {}})

    return fixed_x, I_vals, S_vals, W_vals, Obj_vals, var_solutions, mu, sigma, beta, gamma


if __name__ == "__main__":
    filename = "patients.csv"
    csv_path = os.path.join(".", filename)
    T = list(range(40))
    delta = 20
    tau = 1
    z = 42
    n_patients = 15
    Binary = True
    if not os.path.exists(csv_path):
        print(f"{csv_path} not found. Generating...")
        generate_patient_csv(n_patients=n_patients, delta=delta, T=T, folder=".", filename=filename)
    else:
        print(f"Found: {csv_path}")

    fixed_x, I_vals, S_vals, W_vals, Obj_vals, var_solutions, mu, sigma, beta, gamma = iterative_solver_bias(
        csv_path, T, delta, tau, z, Binary
    )

    offline_I, offline_S, offline_W, offline_Obj = evaluate_offline_objective_over_iterations(
        var_solutions, mu, sigma, beta, gamma, delta, T
    )

    plot_metrics_vs_time(
        t_values=list(range(len(Obj_vals))),
        W_vals=W_vals,
        I_vals=I_vals,
        S_vals=S_vals,
        Obj_vals=Obj_vals,
        offline_I=offline_I,
        offline_S=offline_S,
        offline_W=offline_W,
        offline_Obj=offline_Obj
    )
