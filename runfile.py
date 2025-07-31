'''
%reload_ext autoreload
%autoreload 2
'''

import numpy as np
from tqdm import tqdm
# import stormpy
import subprocess
import pandas as pd
import time
from utils import xy_to_index, index_to_xy
from slipgrid import SlipGrid
from RVI import RVI, compute_derivative
from copy import copy, deepcopy

def plot_grid_layout(grid):
    """
    Plots the layout of a grid environment, including obstacles, initial position, and goal position.
    
    Parameters
    ----------
    grid: An object representing the grid environment. It must have the following attributes:
        - X (int): Number of columns in the grid.
        - Y (int): Number of rows in the grid.
        - obstacles (iterable): Collection of obstacle indices.
        - xy_goal (tuple): Coordinates (x, y) of the goal position.
        - xy_init (tuple): Coordinates (x, y) of the initial position.
    """
    
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    ax.set_xlim(0, grid.X)
    ax.set_ylim(0, grid.Y)
    ax.set_xticks(np.arange(grid.X+1))
    ax.set_yticks(np.arange(grid.Y+1))
    ax.set_yticklabels(np.arange(grid.Y+1)[::-1])  # Reverse Y-axis for correct orientation
    ax.grid(True)

    for obs in grid.obstacles:
        x, y = index_to_xy(obs, grid.Y)
        rect = plt.Rectangle((x, grid.Y - y), 1, -1, color='red', alpha=0.5)
        ax.add_patch(rect)

    x, y = grid.xy_goal
    rect = plt.Rectangle((x, grid.Y - y), 1, -1, color='green', alpha=0.5)
    ax.add_patch(rect)

    x, y = grid.xy_init
    rect = plt.Rectangle((x, grid.Y - y), 1, -1, color='blue', alpha=0.5)
    ax.add_patch(rect)

    # Increase font size 
    plt.xlabel('X', fontsize=32)
    plt.ylabel('Y', fontsize=32)

    # Save to PDF
    # plt.title(f'Grid Layout: {grid.X}x{grid.Y}, Obstacles: {len(grid.obstacles)}')
    fig.tight_layout()
    plt.savefig(f'grid_layout_{grid.X}x{grid.Y}.pdf')

    plt.show()

def count_fraction_besteffort(action_list):
    """
    Calculates the fraction of 'best effort' actions in a given list of actions.

    An action is considered 'best effort' if its string ends with '_be'.
    Actions with the value 'pass' are excluded from the total count.

    Parameters
    ----------
    action_list (list of str): List of action strings.

    Returns
    ----------
    float: Fraction of actions that are 'best effort' (ending with '_be'),
            excluding 'pass' actions. Returns 0 if there are no non-'pass' actions.
    """

    tot_actions = sum(action != 'pass' for action in action_list)
    be_actions = sum(action[-3:] == '_be' for action in action_list)
    fraction_besteffort = be_actions / tot_actions if tot_actions > 0 else 0

    return fraction_besteffort

def run_rvi_instance(grid):
    '''
    Runs a Robust Value Iteration (RVI) instance on the given grid.
    '''
    
    X = grid.X
    Y = grid.Y
    seed = grid.seed

    time0 = time.time()
    V, policy, P_worst, optimal_actions = RVI(grid.rmdp, grid.reward, grid.nr_states, s_init=grid.s_init, gamma=1.0, iters=1000, policy_direction='min', adversary_direction='max')
    rvi_value_minmax = V[grid.s_init]
    rvi_fraction_minmax = count_fraction_besteffort(policy)
    rvi_time_minmax = time.time() - time0

    grid_minmin = deepcopy(grid)
    time0 = time.time()

    # Create a copy of the grid and remove non-optimal actions from the model
    grid_minmin.remove_nonoptimal_actions(optimal_actions)

    # Run RVI for minmin policy
    V, policy, P_worst, optimal_actions = RVI(grid_minmin.rmdp, grid_minmin.reward, grid_minmin.nr_states, s_init=grid_minmin.s_init, gamma=1.0, iters=1000, policy_direction='min', adversary_direction='min')
    rvi_value_minmin = V[grid_minmin.s_init]
    rvi_fraction_minmin = count_fraction_besteffort(policy)
    rvi_time_minmin = time.time() - time0

    grid_derivative = deepcopy(grid)
    time0 = time.time()

    # Create a copy of the grid and remove non-optimal actions from the model
    grid_derivative.remove_nonoptimal_actions(optimal_actions)

    # Compute derivaties for the minmax polic
    derivatives, policy, optimal_actions = compute_derivative(grid_derivative.rmdp, grid_derivative.reward, V, policy, P_worst, -1, s_init=grid_derivative.s_init, s_sink=grid_derivative.s_sink, gamma=1.0, maximize=False)
    rvi_derivative = derivatives[grid_derivative.s_init][policy[grid_derivative.s_init]]
    rvi_fraction_derivative = count_fraction_besteffort(policy)
    rvi_time_derivative = time.time() - time0

    print(f'\n=== RVI results ==============')
    print(f'=== MinMax ===================')
    print(f'Value in initial state:  {np.round(rvi_value_minmax, 2)}')
    print(f'Fraction best-effort:    {np.round(rvi_fraction_minmax, 2)}')
    print(f'Time:                    {np.round(rvi_time_minmax, 2)}')
    print(f'=== MinMin ===================')
    print(f'Value in initial state:  {np.round(rvi_value_minmin, 2)}')
    print(f'Fraction best-effort:    {np.round(rvi_fraction_minmin, 2)}')
    print(f'Time:                    {np.round(rvi_time_minmin, 2)}')
    print(f'=== Derivative================')
    print(f'Derivative in s_init:    {np.round(rvi_derivative, 2)}')
    print(f'Fraction best-effort:    {np.round(rvi_fraction_derivative, 2)}')
    print(f'Time:                    {np.round(rvi_time_derivative, 2)}')
    print(f'==============================')

    return rvi_fraction_minmax, rvi_fraction_minmin, rvi_fraction_derivative, rvi_time_minmax, rvi_time_minmin, rvi_time_derivative

def run_prism_instance(grid):
    '''
    Runs a PRISM instance on the given grid.
    '''

    X = grid.X
    Y = grid.Y
    seed = grid.seed

    # Write the model to a PRISM file
    filename = 'prism/slipgrid_{}_{}_seed={}_minmax.nm'.format(X, Y, seed)
    grid.write_to_prism(filename)

    # Model checking with PRISM (minmax)
    time0 = time.time()
    command = f"/Users/thobad/Documents/Tools/prism/prism/bin/prism {filename} -pf 'Rminmax=? [ F \"goal\" ];' -exportstrat 'prism/policy_minmax.txt' -exportvector 'prism/values_minmax.txt'"
    subprocess.Popen(command, shell=True).wait() 

    optimal_actions = grid.read_prism_output(values='prism/values_minmax.txt', policy='prism/policy_minmax.txt')
    prism_value_minmax = grid.V[grid.s_init]
    prism_fraction_minmax = count_fraction_besteffort(grid.policy)
    prism_time_minmax = time.time() - time0

    grid_minmin = deepcopy(grid)
    time0 = time.time()

    # Remove non-optimal actions from the model
    grid_minmin.remove_nonoptimal_actions(optimal_actions)
    prism_time_minmin0 = time.time() - time0

    # Write the updated model to a new PRISM file
    filename = 'prism/slipgrid_{}_{}_seed={}_minmin.nm'.format(X, Y, seed)
    grid_minmin.write_to_prism(filename)

    # Model checking with PRISM (minmin)
    time0 = time.time()
    command = f"/Users/thobad/Documents/Tools/prism/prism/bin/prism {filename} -pf 'Rminmin=? [ F \"goal\" ];' -exportstrat 'prism/policy_minmin.txt' -exportvector 'prism/values_minmin.txt'"
    subprocess.Popen(command, shell=True).wait()

    grid_minmin.read_prism_output(values='prism/values_minmin.txt', policy='prism/policy_minmin.txt')
    prism_value_minmin = grid_minmin.V[grid_minmin.s_init]
    prism_fraction_minmin = count_fraction_besteffort(grid_minmin.policy)
    prism_time_minmin = time.time() - time0 + prism_time_minmin0

    print(f'\n=== PRISM results =============')
    print(f'=== MinMax ===================')
    print(f'Value in initial state:  {np.round(prism_value_minmax, 2)}')
    print(f'Fraction best-effort:    {np.round(prism_fraction_minmax, 2)}')
    print(f'Time:                    {np.round(prism_time_minmax, 2)}')
    print(f'=== MinMin ===================')
    print(f'Value in initial state:  {np.round(prism_value_minmin, 2)}')
    print(f'Fraction best-effort:    {np.round(prism_fraction_minmin, 2)}')
    print(f'Time:                    {np.round(prism_time_minmin, 2)}')
    print(f'===============================')

    return prism_fraction_minmax, prism_fraction_minmin, prism_time_minmax, prism_time_minmin

if __name__ == "__main__":

    PLOT = False
    if PLOT:
        grid = SlipGrid(X=10, Y=10, B=10, seed=0, p_slip_min=0.1, p_slip_max=0.25, threshold=0.5, verbose=False)
        plot_grid_layout(grid)

    verbose = False
    Xbase = 10     # X cells of the grid
    Ybase = 10     # Y cells of the grid
    Bbase = 10     # Number of obstacles
    sizes = [1,2,3,5,10]
    Ps = [0.0, 0.5, 1.0]    # Probability to define the best-effort action first
    num_seeds = 10
    seeds = np.arange(num_seeds)    # Random seed for reproducibility

    ### PRISM instances ###
    results = {}
    results_stdev = {}

    # Run the instances for each probability P
    for size in sizes:
        # Scale the grid size and number of obstacles
        X = Xbase * size
        Y = Ybase * size
        B = Bbase * size**2
        
        for P in Ps:
            # Initialize the results dictionary for this probability
            size_str = f'{int(X)}$\\times${int(X)}'
            P_str = f'${np.round(P,1)}$'
            key = (size_str, P_str)
            results[key] = {}
            results_stdev[key] = {}

            # Define PRISM results arrays
            prism_be_minmax = np.zeros(len(seeds))
            prism_be_minmin = np.zeros(len(seeds))
            prism_time_minmax = np.zeros(len(seeds))
            prism_time_minmin = np.zeros(len(seeds))
            
            # Run the instances for each random seed
            for i,seed in enumerate(seeds):
                # Seed 7 on the 100x100 grid leads to obstacles that block the path to the goal state completely, so we skip this seed
                if seed == 7 and size == 10:
                    seed = num_seeds

                grid = SlipGrid(X=X, Y=Y, B=B, seed=seed, p_slip_min=0.1, p_slip_max=0.25, threshold=P, verbose=verbose)

                grid.define_IMDP()
                prism_be_minmax[i], prism_be_minmin[i], prism_time_minmax[i], prism_time_minmin[i] = run_prism_instance(grid)

            # Store the results for this probability P
            results[key]['prism-time'] = f'${np.round(np.mean(prism_time_minmax), 1)}$'
            results[key]['prism-perc'] = f'${np.round(np.mean(100*prism_be_minmax), 1)}$'
            results[key]['prism+minmin-time'] = f'${np.round(np.mean(prism_time_minmax + prism_time_minmin), 1)}$'
            results[key]['prism+minmin-perc'] = f'${np.round(np.mean(100*prism_be_minmin), 1)}$'

            # Store the results for this probability P (with standard deviations)
            results_stdev[key]['prism-time'] = f'${np.round(np.mean(prism_time_minmax), 1)} \\pm {np.round(np.std(prism_time_minmax), 1)}$'
            results_stdev[key]['prism-frac'] = f'${np.round(np.mean(prism_be_minmax), 2)} \\pm {np.round(np.std(prism_be_minmax), 2)}$'
            results_stdev[key]['prism+minmin-time'] = f'${np.round(np.mean(prism_time_minmax + prism_time_minmin), 1)} \\pm {np.round(np.std(prism_time_minmax + prism_time_minmin), 1)}$'
            results_stdev[key]['prism+minmin-frac'] = f'${np.round(np.mean(prism_be_minmin), 2)} \\pm {np.round(np.std(prism_be_minmin), 2)}$'

    DF = pd.DataFrame.from_dict(results, orient='index')
    print(DF)
    DF.to_csv('results/slipgrid_prism_results.csv')
    with open('results/slipgrid_prism_results.tex', 'w') as tf:
        tf.write(DF.to_latex())
    DF = pd.DataFrame.from_dict(results_stdev, orient='index')
    DF.to_csv('results/slipgrid_prism_results_stdev.csv')
    with open('results/slipgrid_prism_results_stdev.tex', 'w') as tf:
        tf.write(DF.to_latex())

    verbose = False
    Xbase = 10     # X cells of the grid
    Ybase = 10     # Y cells of the grid
    Bbase = 10     # Number of obstacles
    sizes = [1,2,3]
    Ps = [0.0, 0.5, 1.0]    # Probability to define the best-effort action first
    seeds = np.arange(10)    # Random seed for reproducibility

    ### Robust value iteration instances ###
    results = {}
    results_stdev = {}

    # Run the instances for each probability P
    for size in sizes:
        # Scale the grid size and number of obstacles
        X = Xbase * size
        Y = Ybase * size
        B = Bbase * size**2

        for P in Ps:
            # Initialize the results dictionary for this probability
            size_str = f'{int(X)}$\\times${int(X)}'
            P_str = f'${np.round(P,1)}$'
            key = (size_str, P_str)
            results[key] = {}
            results_stdev[key] = {}

            # Define RVI results arrays
            rvi_be_minmax = np.zeros(len(seeds))
            rvi_be_minmin = np.zeros(len(seeds))
            rvi_be_derivative = np.zeros(len(seeds))
            rvi_time_minmax = np.zeros(len(seeds))
            rvi_time_minmin = np.zeros(len(seeds))
            rvi_time_derivative = np.zeros(len(seeds))
            
            # Run the instances for each random seed
            for i,seed in enumerate(seeds):

                # Run the instance with the given parameters
                grid = SlipGrid(X=X, Y=Y, B=B, seed=seed, p_slip_min=0.1, p_slip_max=0.25, threshold=P, verbose=verbose)
                grid.define_IMDP()
                rvi_be_minmax[i], rvi_be_minmin[i], rvi_be_derivative[i], rvi_time_minmax[i], rvi_time_minmin[i], rvi_time_derivative[i] = run_rvi_instance(grid)

            # Store the results for this probability P
            results[key]['rvi-time'] = f'${np.round(np.mean(rvi_time_minmax), 1)}$'
            results[key]['rvi-perc'] = f'${np.round(np.mean(100*rvi_be_minmax), 1)}$'
            results[key]['rvi+minmin-time'] = f'${np.round(np.mean(rvi_time_minmax + rvi_time_minmin), 1)}$'
            results[key]['rvi+minmin-perc'] = f'${np.round(np.mean(100*rvi_be_minmin), 1)}$'
            results[key]['rvi-derivative-time'] = f'${np.round(np.mean(rvi_time_minmax + rvi_time_derivative), 1)}$'
            results[key]['rvi-derivative-perc'] = f'${np.round(np.mean(100*rvi_be_derivative), 1)}$'

            # Store the results for this probability P (with standard deviations)
            results_stdev[key]['rvi-time'] = f'${np.round(np.mean(rvi_time_minmax), 1)} \\pm {np.round(np.std(rvi_time_minmax), 1)}$'
            results_stdev[key]['rvi-frac'] = f'${np.round(np.mean(rvi_be_minmax), 1)} \\pm {np.round(np.std(rvi_be_minmax), 2)}$'
            results_stdev[key]['rvi+minmin-time'] = f'${np.round(np.mean(rvi_time_minmax + rvi_time_minmin), 1)} \\pm {np.round(np.std(rvi_time_minmax + rvi_time_minmin), 1)}$'
            results_stdev[key]['rvi+minmin-frac'] = f'${np.round(np.mean(rvi_be_minmin), 1)} \\pm {np.round(np.std(rvi_be_minmin), 2)}$'
            results_stdev[key]['rvi-derivative-time'] = f'${np.round(np.mean(rvi_time_minmax + rvi_time_derivative), 1)} \\pm {np.round(np.std(rvi_time_minmax + rvi_time_derivative), 1)}$'
            results_stdev[key]['rvi-derivative-frac'] = f'${np.round(np.mean(rvi_be_derivative), 1)} \\pm {np.round(np.std(rvi_be_derivative), 2)}$'

    DF = pd.DataFrame.from_dict(results, orient='index')
    print(DF)
    DF.to_csv('results/slipgrid_rvi_results.csv')
    with open('results/slipgrid_rvi_results.tex', 'w') as tf:
        tf.write(DF.to_latex())
    DF = pd.DataFrame.from_dict(results_stdev, orient='index')
    DF.to_csv('results/slipgrid_rvi_results_stdev.csv')
    with open('results/slipgrid_rvi_results_stdev.tex', 'w') as tf:
        tf.write(DF.to_latex())