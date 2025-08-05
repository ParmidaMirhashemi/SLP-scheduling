from generate_data import generate_patient_csv
from generate_data import load_patients_from_csv
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# Load or define functions:
# - create_optimization_model
# - visualize_schedule
# - load_patients_from_csv (defined below)

import os

def create_optimization_model(P, T, sigma, kappa, delta, alpha, beta, gamma, fixed_x, t_fixed, tau, Binary = False):
    """
    Create an optimization model for scheduling patients with fixed x values up to t_fixed.
    """
    m = gp.Model()
    
    # Decision variables
    # Decision variables (relaxed to LP: continuous variables in [0,1])
    if Binary == False:
        x = m.addVars(P, T, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
        w = m.addVars(P, T, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="w")
        y = m.addVars(P, T, T, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="y")
        s = m.addVars(P, delta, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="s")
    else:
        x = m.addVars(P, T, vtype=GRB.BINARY,  name="x")
        w = m.addVars(P, T, vtype=GRB.BINARY, name="w")
        y = m.addVars(P, T, T, vtype=GRB.BINARY, name="y")
        s = m.addVars(P, delta, vtype=GRB.BINARY, name="s")


    
    # Fix x values for t <= t_fixed
    for p in P:
        for t in range(t_fixed):
            m.addConstr(x[p, t] == fixed_x[p, t])
    
    for p in P:
        for t in range(t_fixed, t_fixed + tau):
            m.addConstr(x[p, t] >= fixed_x[p, t])
    
    # Capacity constraints
    for t in (T):
        m.addConstr(gp.quicksum(x[p,t] for p in (P)) <= kappa[t])

    # Monotonicity constraints for w
    for p in (P):
        for t in range(sigma[p]-1):
            m.addConstr(w[p,t] - w[p,t+1] <= 0)

    # Linking x and w
    for p in (P):
        for t in range(sigma[p]):
            m.addConstr(x[p,t] - w[p,t] <= 0)
            m.addConstr(w[p,t] - gp.quicksum(x[p,t_] for t_ in range(t+1)) <= 0)

    # y variable constraints
    for p in (P):
        for l in range(2,sigma[p]-1):
            for i in range(1,l+1):
                m.addConstr(y[p,i,l] <= x[p,l+1])
                m.addConstr(y[p,i,l] <= x[p,i-1])

                for k in range(i, l+1):
                    m.addConstr(y[p,i,l] <= 1 - x[p,k])

                m.addConstr(y[p,i,l] >= gp.quicksum(1 - x[p,k] for k in range(i,l+1)) 
                           - (1 - x[p,i-1]) - (1-x[p,l+1]) - (l-i))

    # Assignment constraints
    for p in P:
        m.addConstr(gp.quicksum(x[p, t] for t in range(sigma[p])) == 
                    gp.quicksum(s[p, i] for i in range(1, delta)))
    
    # Monotonicity constraints for s
    for p in P:
        for i in range(1, delta):
            m.addConstr(s[p, i] - s[p, i - 1] <= 0)
    
    # Objective function components
    I = gp.quicksum(gamma[p, i, l] * y[p, i, l] 
                    for p in P
                    for l in range(1, sigma[p] - 1)
                    for i in range(0, l))
    
    S = gp.quicksum(alpha[p, j] * (1 - s[p, j]) 
                    for p in P 
                    for j in range(1, delta))
    
    W = gp.quicksum(beta[p, t] * (1 - w[p, t]) 
                    for p in P 
                    for t in range(sigma[p]))
    
    # Set objective
    m.setObjective(I + S + W, GRB.MINIMIZE)
    
    return m

def visualize_schedule(model, P, T):
    """
    Returns a pandas DataFrame showing the schedule matrix where each cell is 1 if
    patient p is scheduled at time t, and 0 otherwise.
    """
    schedule = pd.DataFrame(0, index=P, columns=T)

    for p in P:
        for t in T:
            var = model.getVarByName(f"x[{p},{t}]")
            if var and var.X > 0.5:
                schedule.at[p, t] = 1

    schedule.index.name = "Patient"
    schedule.columns.name = "Time"
    return schedule


if __name__ == '__main__':

    # Set params
    T = list(range(40))
    delta = 20
    n_patients = 15
    csv_path = "patients.csv"

    # Step 1: Generate data
    csv_path = generate_patient_csv(n_patients, delta, T)

    # Step 2: Load patient parameters
    P, d, nu, sigma_std, sigma = load_patients_from_csv(csv_path, T, delta)

    # Step 3: Build rest of the parameters
    kappa = {t: np.random.randint(3, 6) for t in T}
    beta = {(p, t): np.random.uniform(0.5, 2.0) for p in P for t in T}
    initial_fixed_x = {(p, t): 0 for p in P for t in T}
    t_fixed = 0
    tau = 4

    # Delta and alpha based on true d_p
    Delta = d
    alpha = {(p, j): 50 * (Delta[p] + 1 - j) * (Delta[p] >= j) for p in P for j in range(1, delta)}

    # gamma dimensions depend on sigma[p]
    gamma = {}
    for p in P:
        for l in range(1, sigma[p] - 1):
            for i in range(0, l):
                gamma[p, i, l] = np.random.uniform(0.1, 1.0)

    # Build and solve model
    Binary = False
    model = create_optimization_model(P, T, sigma, kappa, delta, alpha, beta, gamma, initial_fixed_x, t_fixed, tau, Binary)
    model.optimize()

    # Show schedule
    schedule_df = visualize_schedule(model, P, T)
    print(schedule_df)

