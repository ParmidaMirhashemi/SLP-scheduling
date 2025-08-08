import numpy as np
import matplotlib.pyplot as plt
import os
from adaptive_scheduling import iterative_solver, evaluate_offline_objective_over_iterations
from generate_data import generate_patient_csv
from Optimization_model import create_optimization_model
import gurobipy as gp
from adaptive_with_bias import iterative_solver_bias
from gurobipy import GRB
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
from functools import partial
import threading
import time

def compute_optimal_offline_solution(csv_path, T, delta, Binary=False):
    """
    Compute the optimal offline solution with perfect information (true mu values).
    """
    df = pd.read_csv(csv_path)
    P = df["patient_id"].tolist()
    mu = {row["patient_id"]: row["d_p"] for _, row in df.iterrows()}
    sigma = {row["patient_id"]: int(row["nu_p"]) for _, row in df.iterrows()}
    
    # Use true mu values for optimal offline solution
    kappa = {t: 7 for t in T}  # Use average capacity
    beta = {(p, t): (np.sqrt(t*t*t + p) )/ (t + 2)  for p in P for t in T}
    gamma = {(p, i, l): 10 * np.sqrt((i-l+6)*(i-l+5))
             for p in P for l in range(1, max(sigma.values()) - 1) for i in range(0, l)}
    alpha = {(p, j): 50 * (mu[p] + 1 - j) * (mu[p] >= j) for p in P for j in range(1, len(T))}
    
    # No fixed variables for offline optimal
    fixed_x = {(p, t): 0 for p in P for t in T}
    
    # Solve the offline optimization problem
    model = create_optimization_model(P, T, sigma, kappa, delta, alpha, beta, gamma, fixed_x, 0, len(T), Binary)
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        optimal_obj = model.objVal
        return optimal_obj, mu, sigma, beta, gamma
    else:
        raise ValueError("Optimal offline solution is infeasible")

def run_single_seed(seed_info):
    """
    Run a single seed iteration - designed for multiprocessing/multithreading.
    
    Args:
        seed_info: Tuple containing (seed, csv_path, T, delta, tau, Binary, bias, optimal_obj)
    
    Returns:
        Tuple containing results for this seed
    """
    seed, csv_path, T, delta, tau, Binary, bias, optimal_obj = seed_info
    
    try:
        print(f"Thread {threading.current_thread().name}: Running seed {seed}")
        
        if bias:
            fixed_x, I_vals, S_vals, W_vals, Obj_vals, var_solutions, mu, sigma, beta, gamma = iterative_solver_bias(
                csv_path, T, delta, tau, seed, Binary
            )
        else:
            fixed_x, I_vals, S_vals, W_vals, Obj_vals, var_solutions, mu, sigma, beta, gamma = iterative_solver(
                csv_path, T, delta, tau, seed, Binary
            )
        
        # Evaluate offline objective
        offline_I, offline_S, offline_W, offline_Obj = evaluate_offline_objective_over_iterations(
            var_solutions, mu, sigma, beta, gamma, delta, T
        )
        
        # Calculate regret for each iteration
        regret = [obj - optimal_obj for obj in offline_Obj]
        
        print(f"Thread {threading.current_thread().name}: Completed seed {seed}")
        
        return (seed, offline_I, offline_S, offline_W, offline_Obj, regret, True)
        
    except Exception as e:
        print(f"Thread {threading.current_thread().name}: Error in seed {seed}: {e}")
        return (seed, None, None, None, None, None, False)

def run_multi_seed_analysis_threaded(csv_path, T, delta, tau, num_seeds=10, Binary=False, bias=False, max_workers=None):
    """
    Run adaptive scheduling with multiple seeds using multithreading and compute regret.
    
    Args:
        csv_path: Path to patient CSV file
        T: List of time periods
        delta: Maximum number of sessions
        tau: Rolling horizon parameter
        num_seeds: Number of different seeds to run
        Binary: Whether to use binary variables (False for continuous)
        bias: Whether to use bias
        max_workers: Maximum number of threads (None for auto)
    
    Returns:
        Dictionary containing averaged metrics, regret, and individual runs
    """
    
    # First compute the optimal offline solution
    optimal_obj, mu, sigma, beta, gamma = compute_optimal_offline_solution(csv_path, T, delta, Binary)
    print(f"Optimal offline objective: {optimal_obj:.2f}")
    
    all_offline_I = []
    all_offline_S = []
    all_offline_W = []
    all_offline_Obj = []
    all_regret = []
    
    seeds = range(num_seeds)
    
    # Prepare arguments for each seed
    seed_args = [(seed, csv_path, T, delta, tau, Binary, bias, optimal_obj) for seed in seeds]
    
    if max_workers is None:
        max_workers = min(num_seeds, multiprocessing.cpu_count())
    
    print(f"Running {num_seeds} instances with Binary={Binary}, bias={bias} using {max_workers} threads")
    
    # Use ThreadPoolExecutor for I/O bound tasks, ProcessPoolExecutor for CPU bound
    # Since Gurobi optimization is CPU intensive, ProcessPoolExecutor might be better
    # but it requires more memory. Start with ThreadPoolExecutor.
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_seed = {executor.submit(run_single_seed, seed_arg): seed_arg[0] 
                         for seed_arg in seed_args}
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_seed):
            seed = future_to_seed[future]
            try:
                seed, offline_I, offline_S, offline_W, offline_Obj, regret, success = future.result()
                if success:
                    all_offline_I.append(offline_I)
                    all_offline_S.append(offline_S)
                    all_offline_W.append(offline_W)
                    all_offline_Obj.append(offline_Obj)
                    all_regret.append(regret)
                
                completed += 1
                print(f"Completed {completed}/{num_seeds} seeds")
                
            except Exception as e:
                print(f"Error processing seed {seed}: {e}")
    
    if not all_offline_Obj:
        raise ValueError("No successful runs completed")
    
    # Convert to numpy arrays for easier averaging
    all_offline_I = np.array(all_offline_I)
    all_offline_S = np.array(all_offline_S)
    all_offline_W = np.array(all_offline_W)
    all_offline_Obj = np.array(all_offline_Obj)
    all_regret = np.array(all_regret)
    
    # Compute averages and standard deviations
    avg_offline_I = np.mean(all_offline_I, axis=0)
    avg_offline_S = np.mean(all_offline_S, axis=0)
    avg_offline_W = np.mean(all_offline_W, axis=0)
    avg_offline_Obj = np.mean(all_offline_Obj, axis=0)
    avg_regret = np.mean(all_regret, axis=0)
    
    std_offline_I = np.std(all_offline_I, axis=0)
    std_offline_S = np.std(all_offline_S, axis=0)
    std_offline_W = np.std(all_offline_W, axis=0)
    std_offline_Obj = np.std(all_offline_Obj, axis=0)
    std_regret = np.std(all_regret, axis=0)
    
    return {
        'optimal_offline': optimal_obj,
        'averages': {
            'I': avg_offline_I,
            'S': avg_offline_S,
            'W': avg_offline_W,
            'Obj': avg_offline_Obj,
            'regret': avg_regret
        },
        'std_devs': {
            'I': std_offline_I,
            'S': std_offline_S,
            'W': std_offline_W,
            'Obj': std_offline_Obj,
            'regret': std_regret
        },
        'all_runs': {
            'I': all_offline_I,
            'S': all_offline_S,
            'W': all_offline_W,
            'Obj': all_offline_Obj,
            'regret': all_regret
        }
    }

def run_multi_seed_analysis_multiprocess(csv_path, T, delta, tau, num_seeds=10, Binary=False, bias=False, max_workers=None):
    """
    Run adaptive scheduling with multiple seeds using multiprocessing and compute regret.
    Better for CPU-intensive tasks but uses more memory.
    """
    
    # First compute the optimal offline solution
    optimal_obj, mu, sigma, beta, gamma = compute_optimal_offline_solution(csv_path, T, delta, Binary)
    print(f"Optimal offline objective: {optimal_obj:.2f}")
    
    seeds = range(num_seeds)
    seed_args = [(seed, csv_path, T, delta, tau, Binary, bias, optimal_obj) for seed in seeds]
    
    if max_workers is None:
        max_workers = min(num_seeds, multiprocessing.cpu_count())
    
    print(f"Running {num_seeds} instances with Binary={Binary}, bias={bias} using {max_workers} processes")
    
    all_offline_I = []
    all_offline_S = []
    all_offline_W = []
    all_offline_Obj = []
    all_regret = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_seed = {executor.submit(run_single_seed, seed_arg): seed_arg[0] 
                         for seed_arg in seed_args}
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_seed):
            seed = future_to_seed[future]
            try:
                seed, offline_I, offline_S, offline_W, offline_Obj, regret, success = future.result()
                if success:
                    all_offline_I.append(offline_I)
                    all_offline_S.append(offline_S)
                    all_offline_W.append(offline_W)
                    all_offline_Obj.append(offline_Obj)
                    all_regret.append(regret)
                
                completed += 1
                print(f"Completed {completed}/{num_seeds} seeds")
                
            except Exception as e:
                print(f"Error processing seed {seed}: {e}")
    
    if not all_offline_Obj:
        raise ValueError("No successful runs completed")
    
    # Rest of the processing same as threaded version...
    all_offline_I = np.array(all_offline_I)
    all_offline_S = np.array(all_offline_S)
    all_offline_W = np.array(all_offline_W)
    all_offline_Obj = np.array(all_offline_Obj)
    all_regret = np.array(all_regret)
    
    avg_offline_I = np.mean(all_offline_I, axis=0)
    avg_offline_S = np.mean(all_offline_S, axis=0)
    avg_offline_W = np.mean(all_offline_W, axis=0)
    avg_offline_Obj = np.mean(all_offline_Obj, axis=0)
    avg_regret = np.mean(all_regret, axis=0)
    
    std_offline_I = np.std(all_offline_I, axis=0)
    std_offline_S = np.std(all_offline_S, axis=0)
    std_offline_W = np.std(all_offline_W, axis=0)
    std_offline_Obj = np.std(all_offline_Obj, axis=0)
    std_regret = np.std(all_regret, axis=0)
    
    return {
        'optimal_offline': optimal_obj,
        'averages': {
            'I': avg_offline_I,
            'S': avg_offline_S,
            'W': avg_offline_W,
            'Obj': avg_offline_Obj,
            'regret': avg_regret
        },
        'std_devs': {
            'I': std_offline_I,
            'S': std_offline_S,
            'W': std_offline_W,
            'Obj': std_offline_Obj,
            'regret': std_regret
        },
        'all_runs': {
            'I': all_offline_I,
            'S': all_offline_S,
            'W': all_offline_W,
            'Obj': all_offline_Obj,
            'regret': all_regret
        }
    }

def plot_averaged_metrics(results, Binary=False):
    """
    Plot averaged offline metrics with error bars showing standard deviation.
    """
    averages = results['averages']
    std_devs = results['std_devs']
    
    t_values = list(range(len(averages['I'])))
    
    plt.figure(figsize=(12, 8))
    
    # Plot averages with error bars
    plt.errorbar(t_values, averages['I'], yerr=std_devs['I'], 
                label="I (Interruption Cost)", linestyle="-", marker='o', markersize=4, capsize=3)
    plt.errorbar(t_values, averages['S'], yerr=std_devs['S'], 
                label="S (Shortage Cost)", linestyle="-", marker='s', markersize=4, capsize=3)
    plt.errorbar(t_values, averages['W'], yerr=std_devs['W'], 
                label="W (Waiting Cost)", linestyle="-", marker='^', markersize=4, capsize=3)
    plt.errorbar(t_values, averages['Obj'], yerr=std_devs['Obj'], 
                label="Total Objective", linestyle="--", marker='d', markersize=4, capsize=3, linewidth=2)
    
    plt.xlabel("t_fixed (Time Step)")
    plt.ylabel("Objective Component Value")
    variable_type = "Binary" if Binary else "Continuous"
    plt.title(f"Average Offline-Evaluated Objective Components ({variable_type} Variables)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_individual_runs(results, Binary=False):
    """
    Plot all individual runs with the average highlighted.
    """
    all_runs = results['all_runs']
    averages = results['averages']
    
    t_values = list(range(len(averages['I'])))
    
    plt.figure(figsize=(12, 8))
    
    # Plot individual runs with transparency
    for i in range(len(all_runs['Obj'])):
        plt.plot(t_values, all_runs['Obj'][i], 'gray', alpha=0.3, linewidth=1)
    
    # Plot average with thick line
    plt.plot(t_values, averages['Obj'], 'red', linewidth=3, label="Average Total Objective")
    
    plt.xlabel("t_fixed (Time Step)")
    plt.ylabel("Total Objective Value")
    variable_type = "Binary" if Binary else "Continuous"
    plt.title(f"Individual Runs vs Average Total Objective ({variable_type} Variables)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Parameters
    filename = "patients.csv"
    csv_path = os.path.join(".", filename)
    T = list(range(40))
    delta = 20
    tau = 1
    n_patients = 15
    num_seeds = 15
    Binary = False  # Use continuous variables
    max_workers = 8  # Adjust based on your CPU cores

    # Generate patient data if it doesn't exist
    if not os.path.exists(csv_path):
        print(f"{csv_path} not found. Generating...")
        generate_patient_csv(n_patients=n_patients, delta=delta, T=T, folder=".", filename=filename)
    else:
        print(f"Found: {csv_path}")

    # Choose between threading or multiprocessing
    use_multiprocessing = True  # Set to False to use threading instead
    
    if use_multiprocessing:
        print("Using multiprocessing...")
        analysis_func = run_multi_seed_analysis_multiprocess
    else:
        print("Using multithreading...")
        analysis_func = run_multi_seed_analysis_threaded

    # Measure execution time
    start_time = time.time()

    # Run multi-seed analysis for both bias and no-bias
    print(f"\nRunning multi-seed analysis with {num_seeds} seeds (no bias)...")
    results_no_bias = analysis_func(csv_path, T, delta, tau, num_seeds, Binary, bias=False, max_workers=max_workers)

    print(f"\nRunning multi-seed analysis with {num_seeds} seeds (with bias)...")
    results_with_bias = analysis_func(csv_path, T, delta, tau, num_seeds, Binary, bias=True, max_workers=max_workers)

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

    # Print summary statistics
    print(f"\nSummary Statistics (Binary={Binary}):")
    print(f"Final Average Total Objective (no bias): {results_no_bias['averages']['Obj'][-1]:.2f} ± {results_no_bias['std_devs']['Obj'][-1]:.2f}")
    print(f"Final Average Total Objective (with bias): {results_with_bias['averages']['Obj'][-1]:.2f} ± {results_with_bias['std_devs']['Obj'][-1]:.2f}")

    # Plot comparison of average total objective
    t_values = list(range(len(results_no_bias['averages']['Obj'])))
    plt.figure(figsize=(10, 6))
    plt.plot(t_values, results_no_bias['averages']['Obj'], label="No Bias", color="blue")
    plt.fill_between(t_values,
                     results_no_bias['averages']['Obj'] - results_no_bias['std_devs']['Obj'],
                     results_no_bias['averages']['Obj'] + results_no_bias['std_devs']['Obj'],
                     color="blue", alpha=0.2)
    plt.plot(t_values, results_with_bias['averages']['Obj'], label="With Bias", color="red")
    plt.fill_between(t_values,
                     results_with_bias['averages']['Obj'] - results_with_bias['std_devs']['Obj'],
                     results_with_bias['averages']['Obj'] + results_with_bias['std_devs']['Obj'],
                     color="red", alpha=0.2)
    plt.xlabel("t_fixed (Time Step)")
    plt.ylabel("Average Total Objective Value")
    plt.title(f"Comparison of Average Total Objective (n_patients={n_patients}, seeds={num_seeds})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("Analysis complete!")