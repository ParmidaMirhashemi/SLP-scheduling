# SLP-Scheduling

A Speech-Language Pathology (SLP) scheduling optimization system that uses mathematical programming to efficiently allocate therapy sessions for patients while considering constraints such as capacity, patient needs, and discharge timing.

## Overview

This project implements an adaptive scheduling algorithm for SLP services using optimization models. The system can handle dynamic patient requirements and provides both binary and continuous optimization approaches.

## Files

- **`Optimization_model.py`** - Core optimization model implementation using Gurobi solver
- **`adaptive_scheduling.py`** - Adaptive iterative scheduling algorithm with offline evaluation
- **`generate_data.py`** - Patient data generation and CSV loading utilities
- **`patients.csv`** - Sample patient data with session requirements and discharge times

## Features

- **Mathematical Optimization**: Uses Gurobi optimizer for scheduling decisions
- **Adaptive Scheduling**: Iterative solver that adjusts to changing patient needs
- **Multiple Objectives**: Minimizes interruption costs, shortage penalties, and waiting costs
- **Binary/Continuous Variables**: Supports both binary and relaxed LP formulations
- **Data Generation**: Automated patient data generation for testing scenarios
- **Visualization**: Schedule visualization and performance metrics plotting

## Requirements

- Python 3.x
- pandas
- numpy
- gurobipy (Gurobi Optimizer)
- matplotlib
- dataclasses

## Usage

### Basic Optimization Model
```python
from Optimization_model import create_optimization_model, visualize_schedule
from generate_data import generate_patient_csv, load_patients_from_csv

# Generate or load patient data
n_patients = 15
T = list(range(40))  # Time periods
delta = 20  # Maximum sessions

# Create and solve model
model = create_optimization_model(P, T, sigma, kappa, delta, alpha, beta, gamma, fixed_x, t_fixed, tau)
model.optimize()

# Visualize results
schedule_df = visualize_schedule(model, P, T)
print(schedule_df)
```

### Adaptive Scheduling
```python
from adaptive_scheduling import iterative_solver, evaluate_offline_objective_over_iterations

# Run adaptive scheduling
fixed_x, I_vals, S_vals, W_vals, Obj_vals, var_solutions, mu, sigma, beta, gamma = iterative_solver(
    csv_path="patients.csv", 
    T=list(range(40)), 
    delta=20, 
    tau=1, 
    z=42
)

# Evaluate offline performance
offline_I, offline_S, offline_W, offline_Obj = evaluate_offline_objective_over_iterations(
    var_solutions, mu, sigma, beta, gamma, delta, T
)
```

## Model Parameters

- **P**: Set of patients
- **T**: Set of time periods
- **sigma[p]**: Latest possible scheduling time for patient p
- **kappa[t]**: Capacity at time t
- **delta**: Maximum number of sessions
- **alpha[p,j]**: Shortage penalty for patient p missing j sessions
- **beta[p,t]**: Waiting cost for patient p at time t
- **gamma[p,i,l]**: Interruption cost for patient p with gap from i to l

## Decision Variables

- **x[p,t]**: Binary/continuous variable indicating if patient p is scheduled at time t
- **w[p,t]**: Binary/continuous variable for cumulative scheduling up to time t
- **s[p,j]**: Binary/continuous variable for session assignment
- **y[p,i,l]**: Binary/continuous variable for interruption gaps

## Getting Started

1. Install required dependencies:
   ```bash
   pip install pandas numpy gurobipy matplotlib
   ```

2. Generate sample data:
   ```python
   python generate_data.py
   ```

3. Run the optimization model:
   ```python
   python Optimization_model.py
   ```

4. Run adaptive scheduling:
   ```python
   python adaptive_scheduling.py
   ```

## License

This project is available for research and educational purposes.