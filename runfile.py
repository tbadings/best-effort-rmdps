import numpy as np
from tqdm import tqdm
# import stormpy
import subprocess
import pandas as pd
import time
from utils import xy_to_index, index_to_xy
from slipgrid import SlipGrid
from RVI import RVI

def plot_grid_layout(X, Y, obstacles=None):
    """Plot the grid layout with obstacles."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.5, X - 0.5)
    ax.set_ylim(-0.5, Y - 0.5)
    ax.set_xticks(np.arange(X))
    ax.set_yticks(np.arange(Y))
    ax.grid(True)

    if obstacles is not None:
        for obs in obstacles:
            x, y = index_to_xy(obs, Y)
            rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='red', alpha=0.5)
            ax.add_patch(rect)

    plt.show()

def count_fraction_besteffort(action_list):

    tot_actions = sum(action != 'pass' for action in action_list)
    be_actions = sum(action[-3:] == '_be' for action in action_list)
    fraction_besteffort = be_actions / tot_actions if tot_actions > 0 else 0

    return fraction_besteffort

def run_rvi_instance(grid):
    
    X = grid.X
    Y = grid.Y
    seed = grid.seed

    time0 = time.time()

    V, policy, P_worst, optimal_actions = RVI(grid.rmdp, grid.reward, grid.nr_states, s_init=grid.s_init, gamma=1.0, iters=100, policy_direction='min', adversary_direction='max')
    rvi_value_minmax = V[grid.s_init]
    rvi_fraction_minmax = count_fraction_besteffort(policy)
    rvi_time = time.time() - time0
    
    # Remove non-optimal actions from the model
    grid.remove_nonoptimal_actions(optimal_actions)

    V, policy, P_worst, optimal_actions = RVI(grid.rmdp, grid.reward, grid.nr_states, s_init=grid.s_init, gamma=1.0, iters=100, policy_direction='min', adversary_direction='min')
    rvi_fraction_minmin = count_fraction_besteffort(policy)
    rvi_time_double = time.time() - time0

    print(f'\n===== RVI results (minmax) =====')
    print(f'Value in initial state:  {np.round(rvi_value_minmax, 2)}')
    print(f'Fraction best-effort:    {np.round(rvi_fraction_minmax, 2)}')
    print(f'Time:                    {np.round(rvi_time, 2)}')
    print(f'===== RVI results (minmin) =====')
    print(f'Value in initial state:  {np.round(V[grid.s_init], 2)}')
    print(f'Fraction best-effort:    {np.round(rvi_fraction_minmin, 2)}')
    print(f'Time:                    {np.round(rvi_time_double, 2)}')
    print(f'================================')

    return rvi_fraction_minmax, rvi_fraction_minmin, rvi_time, rvi_time_double

def run_prism_instance(grid):

    X = grid.X
    Y = grid.Y
    seed = grid.seed

    time0 = time.time()

    # Write the model to a PRISM file
    filename = 'prism/slipgrid_{}_{}_seed={}_minmax.nm'.format(X, Y, seed)
    grid.write_to_prism(filename)

    # Model checking with PRISM (minmax)
    command = f"/Users/thobad/Documents/Tools/prism/prism/bin/prism {filename} -pf 'Rminmax=? [ F \"goal\" ];' -exportstrat 'prism/policy_minmax.txt' -exportvector 'prism/values_minmax.txt'"
    subprocess.Popen(command, shell=True).wait() 

    optimal_actions = grid.read_prism_output(values='prism/values_minmax.txt', policy='prism/policy_minmax.txt')
    prism_value_minmax = grid.V[grid.s_init]
    prism_fraction_minmax = count_fraction_besteffort(grid.policy)
    prism_time = time.time() - time0

    # Remove non-optimal actions from the model
    grid.remove_nonoptimal_actions(optimal_actions)
    # Write the updated model to a new PRISM file
    filename = 'prism/slipgrid_{}_{}_seed={}_minmin.nm'.format(X, Y, seed)
    grid.write_to_prism(filename)

    # Model checking with PRISM (minmin)
    command = f"/Users/thobad/Documents/Tools/prism/prism/bin/prism {filename} -pf 'Rminmin=? [ F \"goal\" ];' -exportstrat 'prism/policy_minmin.txt' -exportvector 'prism/values_minmin.txt'"
    subprocess.Popen(command, shell=True).wait()

    grid.read_prism_output(values='prism/values_minmin.txt', policy='prism/policy_minmin.txt')
    prism_fraction_minmin = count_fraction_besteffort(grid.policy)
    prism_time_double = time.time() - time0

    print(f'\n===== PRISM results (minmax) =====')
    print(f'Value in initial state:  {np.round(prism_value_minmax, 2)}')
    print(f'Fraction best-effort:    {np.round(prism_fraction_minmax, 2)}')
    print(f'Time:                    {np.round(prism_time, 2)}')
    print(f'===== PRISM results (minmin) =====')
    print(f'Value in initial state:  {np.round(grid.V[grid.s_init], 2)}')
    print(f'Fraction best-effort:    {np.round(prism_fraction_minmin, 2)}')
    print(f'Time:                    {np.round(prism_time_double, 2)}')
    print(f'===============================')

    return prism_fraction_minmax, prism_fraction_minmin, prism_time, prism_time_double

verbose = False
Xbase = 10     # X cells of the grid
Ybase = 10     # Y cells of the grid
Bbase = 10     # Number of obstacles
sizes = [1]
Ps = [0.0, 0.5, 1.0] #[0.0, 0.5, 1.0]    # Probability to define the best-effort action first
seeds = np.arange(2)    # Random seed for reproducibility
results = {}

# Run the instances for each probability P
for size in sizes:
    # Scale the grid size and number of obstacles
    X = Xbase * size
    Y = Ybase * size
    B = Bbase * size**2

    for P in Ps:
        # Initialize the results dictionary for this probability
        results[(size, P)] = {}

        # Define PRISM results arrays
        prism_be_minmax = np.zeros(len(seeds))
        prism_be_minmin = np.zeros(len(seeds))
        prism_time = np.zeros(len(seeds))
        prism_time_double = np.zeros(len(seeds))
        
        # Run the instances for each random seed
        for i,seed in enumerate(seeds):

            grid = SlipGrid(X=X, Y=Y, B=B, seed=seed, p_slip_min=0.1, p_slip_max=0.25, p_to_sink=0.1, threshold=P, verbose=verbose)
            grid.define_IMDP()
            prism_be_minmax[i], prism_be_minmin[i], prism_time[i], prism_time_double[i] = run_prism_instance(grid)

        # Store the results for this probability P
        results[(size, P)]['prism-time'] = f'${np.round(np.mean(prism_time), 1)} \pm {np.round(np.std(prism_time), 1)}$'
        results[(size, P)]['prism-frac'] = f'${np.round(np.mean(prism_be_minmax), 1)} \pm {np.round(np.std(prism_be_minmax), 1)}$'
        results[(size, P)]['ours-time'] = f'${np.round(np.mean(prism_time_double), 1)} \pm {np.round(np.std(prism_time_double), 1)}$'
        results[(size, P)]['ours-frac'] = f'${np.round(np.mean(prism_be_minmin), 1)} \pm {np.round(np.std(prism_be_minmin), 1)}$'

DF = pd.DataFrame.from_dict(results, orient='index')
print(DF)
DF.to_csv('results/slipgrid_prism_results.csv')
with open('results/slipgrid_prism_results.tex', 'w') as tf:
     tf.write(DF.to_latex())

# Run the instances for each probability P
for size in sizes:
    # Scale the grid size and number of obstacles
    X = Xbase * size
    Y = Ybase * size
    B = Bbase * size**2

    for P in Ps:
        # Initialize the results dictionary for this probability
        results[(size, P)] = {}

        # Define RVI results arrays
        rvi_be_minmax = np.zeros(len(seeds))
        rvi_be_minmin = np.zeros(len(seeds))
        rvi_time = np.zeros(len(seeds))
        rvi_time_double = np.zeros(len(seeds))
        
        # Run the instances for each random seed
        for i,seed in enumerate(seeds):

            # Run the instance with the given parameters
            grid = SlipGrid(X=X, Y=Y, B=B, seed=seed, p_slip_min=0.1, p_slip_max=0.25, p_to_sink=0.1, threshold=P, verbose=verbose)
            grid.define_IMDP()
            rvi_be_minmax[i], rvi_be_minmin[i], rvi_time[i], rvi_time_double[i] = run_rvi_instance(grid)

        # Store the results for this probability P
        results[(size, P)]['rvi-time'] = f'${np.round(np.mean(rvi_time), 1)} \pm {np.round(np.std(rvi_time), 1)}$'
        results[(size, P)]['rvi-frac'] = f'${np.round(np.mean(rvi_be_minmax), 1)} \pm {np.round(np.std(rvi_be_minmax), 1)}$'
        results[(size, P)]['ours-time'] = f'${np.round(np.mean(rvi_time_double), 1)} \pm {np.round(np.std(rvi_time_double), 1)}$'
        results[(size, P)]['ours-frac'] = f'${np.round(np.mean(rvi_be_minmin), 1)} \pm {np.round(np.std(rvi_be_minmin), 1)}$'

DF = pd.DataFrame.from_dict(results, orient='index')
print(DF)
DF.to_csv('results/slipgrid_rvi_results.csv')
with open('results/slipgrid_rvi_results.tex', 'w') as tf:
     tf.write(DF.to_latex())

# Generate IMDP with Storm
# nr_states, obstacles = define_slipgrids(X=X, Y=Y, B=B, seed=0, p_slip_min=0.1, p_slip_max=0.25, p_to_sink=0.1, threshold = P, verbose=True)

# Run PRISM command
# /Users/thobad/Documents/Tools/prism/prism/bin/prism prism/slipgrid_5_5_seed=0.nm -pf 'Rminmax=? [ F "goal" ];' -exportstrat 'prism/policy.txt' -exportvector 'prism/values.txt'