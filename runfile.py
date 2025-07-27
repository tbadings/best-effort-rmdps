from tabnanny import verbose
import numpy as np
from tqdm import tqdm
import stormpy
import subprocess
import pandas as pd
import time

def xy_to_index(x, y, X, Y):
    """Convert (x, y) coordinates to a single index."""

    # Grid looks like this:
    # 0 1 2 3 4
    # 5 6 7 8 9
    # ... and so on

    # Ensure x and y are within bounds
    y = min(max(0, y), Y - 1)
    x = min(max(0, x), X - 1)

    return x * X + y

def index_to_xy(index, Y):
    """Convert a single index back to (x, y) coordinates."""

    # Calculate y coordinate
    y = index % Y

    # Calculate x coordinate
    x = index // Y

    return x, y

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

class SlipGrid:
    def __init__(self, X=5, Y=5, B=10, seed=0, p_slip_min=0.1, p_slip_max=0.2, p_to_sink=0.1, threshold=0.5, verbose=False):
        self.X = X
        self.Y = Y
        self.B = B
        self.seed = seed
        self.p_slip_min = p_slip_min
        self.p_slip_max = p_slip_max
        self.p_to_sink = p_to_sink
        self.threshold = threshold
        self.verbose = verbose

        # Set the random seed for reproducibility
        np.random.seed(seed)
        
        # Define the number of states and choices based on X and Y
        self.nr_states = X * Y

        # Randomly select O unique obstacles indexes (avoid choosing the initial and goal states)
        self.obstacles = np.random.choice(self.nr_states - 2, size=B, replace=False) + 1
        print(f' - Selected {len(self.obstacles)} obstacles: {self.obstacles}')

    def define_IMDP(self):

        self.imdp = {}
        self.reward = {}
        self.p_worst = {}
        self.nr_choices = 0

        # Define transition probability function
        for x in range(X):
            for y in range(Y):
                if self.verbose:
                    print(f'-- Processing state ({x}, {y})')

                # Calculate the state index from (x, y) coordinates
                i = xy_to_index(x, y, X, Y)

                self.imdp[i] = {}
                self.reward[i] = {}

                delta_min = np.round(np.random.uniform(-0.05, 0.05), 3)
                delta_max = np.round(np.random.uniform(-0.05, 0.05), 3)      

                if x == X - 1 and y == Y - 1:  # Goal state
                    self.p_worst[i] = [1]

                    self.imdp[i]['pass'] = {}
                    self.imdp[i]['pass'][i] = [1.0, 1.0]
                    self.reward[i]['pass'] = 0
                    self.nr_choices += 1

                    if self.verbose:
                        print(f' - State ({x}, {y}) is the goal state, adding pass action.')
                elif i in self.obstacles:
                    self.p_worst[i] = [1]

                    self.imdp[i]['pass'] = {}
                    self.imdp[i]['pass'][0] = [1.0, 1.0]
                    self.reward[i]['pass'] = 0
                    self.nr_choices += 1

                    if self.verbose:
                        print(f' - State ({x}, {y}) is an obstacle, adding pass action to sink state.')
                else:
                    self.p_worst[i] = [1 - (self.p_slip_max + delta_max), 
                                   self.p_slip_max + delta_max]

                    # Define the actions based on the position in the grid
                    if x > 0: # If x > 0, can move left
                        self.nr_choices += 2
                        self.reward[i]['left'] = 1
                        self.reward[i]['left_be'] = 1

                        if np.random.rand() >= self.threshold:                            
                            self.imdp[i]['left'] = {
                                xy_to_index(x - 1, y, X, Y): [1 - (self.p_slip_max + delta_max), 1 - (self.p_slip_max + delta_max)],
                                i: [self.p_slip_max + delta_max, self.p_slip_max + delta_max]
                            }
                            self.imdp[i]['left_be'] = {
                                xy_to_index(x - 1, y, X, Y): [1 - (self.p_slip_max + delta_max), 1 - (self.p_slip_min + delta_min)],
                                i: [self.p_slip_min + delta_min, self.p_slip_max + delta_max]
                            }
                        else:
                            self.imdp[i]['left_be'] = {
                                xy_to_index(x - 1, y, X, Y): [1 - (self.p_slip_max + delta_max), 1 - (self.p_slip_min + delta_min)],
                                i: [self.p_slip_min + delta_min, self.p_slip_max + delta_max]
                            }
                            self.imdp[i]['left'] = {
                                xy_to_index(x - 1, y, X, Y): [1 - (self.p_slip_max + delta_max), 1 - (self.p_slip_max + delta_max)],
                                i: [self.p_slip_max + delta_max, self.p_slip_max + delta_max]
                            }

                    if x < X - 1: # If x < X-1, can move right
                        self.nr_choices += 2
                        self.reward[i]['right'] = 1
                        self.reward[i]['right_be'] = 1

                        if np.random.rand() >= self.threshold:
                            self.imdp[i]['right'] = {
                                xy_to_index(x + 1, y, X, Y): [1 - (self.p_slip_max + delta_max), 1 - (self.p_slip_max + delta_max)],
                                i: [self.p_slip_max + delta_max, self.p_slip_max + delta_max]
                            }
                            self.imdp[i]['right_be'] = {
                                xy_to_index(x + 1, y, X, Y): [1 - (self.p_slip_max + delta_max), 1 - (self.p_slip_min + delta_min)],
                                i: [self.p_slip_min + delta_min, self.p_slip_max + delta_max]
                            }
                        else:
                            self.imdp[i]['right_be'] = {
                                xy_to_index(x + 1, y, X, Y): [1 - (self.p_slip_max + delta_max), 1 - (self.p_slip_min + delta_min)],
                                i: [self.p_slip_min + delta_min, self.p_slip_max + delta_max]
                            }
                            self.imdp[i]['right'] = {
                                xy_to_index(x + 1, y, X, Y): [1 - (self.p_slip_max + delta_max), 1 - (self.p_slip_max + delta_max)],
                                i: [self.p_slip_max + delta_max, self.p_slip_max + delta_max]
                            }
                        
                    if y > 0: # If y > 0, can move up
                        self.nr_choices += 2
                        self.reward[i]['up'] = 1
                        self.reward[i]['up_be'] = 1

                        if np.random.rand() >= self.threshold:
                            self.imdp[i]['up'] = {
                                xy_to_index(x, y - 1, X, Y): [1 - (self.p_slip_max + delta_max), 1 - (self.p_slip_max + delta_max)],
                                i: [self.p_slip_max + delta_max, self.p_slip_max + delta_max]
                            }
                            self.imdp[i]['up_be'] = {
                                xy_to_index(x, y - 1, X, Y): [1 - (self.p_slip_max + delta_max), 1 - (self.p_slip_min + delta_min)],
                                i: [self.p_slip_min + delta_min, self.p_slip_max + delta_max]
                            }
                        else:
                            self.imdp[i]['up_be'] = {
                                xy_to_index(x, y - 1, X, Y): [1 - (self.p_slip_max + delta_max), 1 - (self.p_slip_min + delta_min)],
                                i: [self.p_slip_min + delta_min, self.p_slip_max + delta_max]
                            }
                            self.imdp[i]['up'] = {
                                xy_to_index(x, y - 1, X, Y): [1 - (self.p_slip_max + delta_max), 1 - (self.p_slip_max + delta_max)],
                                i: [self.p_slip_max + delta_max, self.p_slip_max + delta_max]
                            }

                    if y < Y - 1: # If y < Y-1, can move down
                        self.nr_choices += 2
                        self.reward[i]['down'] = 1
                        self.reward[i]['down_be'] = 1

                        if np.random.rand() >= self.threshold:
                            self.imdp[i]['down'] = {
                                xy_to_index(x, y + 1, X, Y): [1 - (self.p_slip_max + delta_max), 1 - (self.p_slip_max + delta_max)],
                                i: [self.p_slip_max + delta_max, self.p_slip_max + delta_max]
                            }
                            self.imdp[i]['down_be'] = {
                                xy_to_index(x, y + 1, X, Y): [1 - (self.p_slip_max + delta_max), 1 - (self.p_slip_min + delta_min)],
                                i: [self.p_slip_min + delta_min, self.p_slip_max + delta_max]
                            }
                        else:
                            self.imdp[i]['down_be'] = {
                                xy_to_index(x, y + 1, X, Y): [1 - (self.p_slip_max + delta_max), 1 - (self.p_slip_min + delta_min)],
                                i: [self.p_slip_min + delta_min, self.p_slip_max + delta_max]
                            }
                            self.imdp[i]['down'] = {
                                xy_to_index(x, y + 1, X, Y): [1 - (self.p_slip_max + delta_max), 1 - (self.p_slip_max + delta_max)],
                                i: [self.p_slip_max + delta_max, self.p_slip_max + delta_max]
                            }

        self.rewards = {
            'left': 1,
            'left_be': 1,
            'right': 1,
            'right_be': 1,
            'up': 1,
            'up_be': 1,
            'down': 1,
            'down_be': 1,
            'pass': 0
        }

    def write_to_prism(self, filename):

        Q =  ['mdp\n',
          'module grid',
          f'    x : [0..{(X-1)}];',
          f'    y : [0..{(Y-1)}];']
    
        for s,v in self.imdp.items():
            for a,w in v.items():
                # Compute the coordinates from the state index
                x,y = index_to_xy(s, Y)

                str = [f'[{p[0]}, {p[1]}]: (x\'={index_to_xy(ss, Y)[0]}) & (y\'={index_to_xy(ss, Y)[1]})' for ss,p in w.items()]
                str = ' + '.join(str)

                # Add the transition to the model
                Q += [f'    [{a}] x={x} & y={y} -> {str};']

        Q += ['endmodule\n',
            '// initial states',
            'init true endinit\n',
            '// reward structure (number of steps to reach the target)',
            'rewards']
        for action, r in self.rewards.items():
            Q += [f'    [{action}] true : {r};']
        Q += ['endrewards\n',
            f'label "goal" = x={X-1} & y={Y-1};'
            ]

        # Print to file
        with open(f'{filename}', 'w') as fp:
            fp.write('\n'.join(Q))
            
        print('Exported model with name "{}"'.format(filename))
        
    def write_to_storm(self):
        # Create a builder for the sparse interval matrix
        builder = stormpy.IntervalSparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False,
                                                has_custom_row_grouping=True, row_groups=0)    
        
        choice_labeling = stormpy.storage.ChoiceLabeling(self.nr_choices)

        # Add labels for the actions; two types of actions: normal and best-effort
        choice_labeling.add_label('pass')
        choice_labeling.add_label('up')
        choice_labeling.add_label('up_be')
        choice_labeling.add_label('down')
        choice_labeling.add_label('down_be')
        choice_labeling.add_label('left')
        choice_labeling.add_label('left_be')
        choice_labeling.add_label('right')
        choice_labeling.add_label('right_be')

        row = 0
        reward_per_choice = []

        for s,v in self.imdp.items():
            for a,w in v.items():

                # Add a new row group for each state
                builder.new_row_group(row)

                # Add the action label to the choice labeling
                choice_labeling.add_label_to_choice(str(a), row)

                if a == 'pass':
                    reward_per_choice += [0]
                else:
                    reward_per_choice += [1]

                for ss,p in w.items():
                    print(f' - Adding transition from state {s} to state {ss} with action {a} and probability {p}')

                    # Add the next transition to the builder
                    builder.add_next_value(row, ss, stormpy.pycarl.Interval(p[0], p[1]))

                row += 1


        # Add state labeling
        state_labeling = stormpy.storage.StateLabeling(self.nr_states)
        state_labeling.add_label('init')
        state_labeling.add_label('goal')
        state_labeling.add_label_to_state('init', xy_to_index(0, 0, X, Y))
        state_labeling.add_label_to_state('goal', xy_to_index(X - 1, Y - 1, X, Y))

        # Add the reward model
        reward_models = {}
        action_reward = [stormpy.pycarl.Interval(r,r) for r in reward_per_choice]
        reward_models["reward"] = stormpy.SparseIntervalRewardModel(optional_state_action_reward_vector=action_reward)

        matrix = builder.build()

        components = stormpy.SparseIntervalModelComponents(transition_matrix=matrix, state_labeling=state_labeling)
        components.choice_labeling = choice_labeling
        imdp = stormpy.storage.SparseIntervalMdp(components)

        if verbose:
            print('- IMDP created.')

        prop = stormpy.parse_properties('Rmin=? [F "goal"]')[0]
        env = stormpy.Environment()
        env.solver_environment.minmax_solver_environment.method = stormpy.MinMaxMethod.value_iteration
                   
        if verbose:
            print('- Property parsed and environment set up.')

        task = stormpy.CheckTask(prop.raw_formula, only_initial_states=False)
        task.set_produce_schedulers()
        task.set_robust_uncertainty(True)

        if verbose:
            print('- Task created.')

        result_generator = stormpy.check_interval_mdp(imdp, task, env)

        if verbose:
            print('Model checked.')

        self.imdp = imdp
        self.result_generator = result_generator
            
    def read_prism_output(self, values, policy):
        """Read the output from a PRISM file."""
        with open(values, 'r') as f:
            lines = f.readlines()

            self.V = np.array([float(line.split('\n')[0]) for line in lines])

        with open(policy, 'r') as f:
            lines = f.readlines()

            self.policy = [line.split('=')[1].split('\n')[0] for line in lines]

        tot_actions = sum(action != 'pass' for action in self.policy)
        be_actions = sum(action[-3:] == '_be' for action in self.policy)
        self.fraction_besteffort = be_actions / tot_actions if tot_actions > 0 else 0

        # Assess which actions are optimal per state
        self.optimal_actions = {}
        for s,v in self.imdp.items():
            self.optimal_actions[s] = []
            x,y = index_to_xy(s, self.Y)
            for a,w in v.items():
                value = self.reward[s][a] + np.sum([self.p_worst[s][j] * self.V[ss] for j,ss in enumerate(w.keys())])

                if s == 3:
                    print('value in state 3, action:', a, 'is:', value)

                if np.abs(value - self.V[s]) < 1e-4:
                    if self.verbose:
                        print(f' - Value for action {a} in state {s} ({x},{y}) is {value} (optimal)')
                    self.optimal_actions[s] += [a]
                else:
                    if self.verbose:
                        print(f' - Value for action {a} in state {s} ({x},{y}) is {value} (not optimal)')

    def remove_nonoptimal_actions(self):
        
        for s,v in self.imdp.items():
            keys = list(v.keys())
            for action in keys:
                if action not in self.optimal_actions[s]:
                    print(f' - Removing non-optimal action {action} from state {s}')
                    del self.imdp[s][action]
                    del self.reward[s][action]
                    self.nr_choices -= 1

        # Check if some actions are not enabled in any state anymore
        for action in list(self.rewards.keys()):
            if not any([action in enabled_actions.keys() for enabled_actions in self.imdp.values()]):
                print(f' - Removing action {action} from rewards as it is not enabled in any state anymore')
                del self.rewards[action]

def run_instance(X, Y, B, P, seed=0, storm=True, prism=True, verbose=False):

    time0 = time.time()

    grid = SlipGrid(X=X, Y=Y, B=B, seed=seed, p_slip_min=0.1, p_slip_max=0.25, p_to_sink=0.1, threshold=P, verbose=verbose)
    grid.define_IMDP()

    # Plot the grid layout with obstacles
    # plot_grid_layout(X, Y, grid.obstacles)

    if prism:
        # Write the model to a PRISM file
        filename = 'prism/slipgrid_{}_{}_seed={}_minmax.nm'.format(X, Y, seed)
        grid.write_to_prism(filename)

        # Model checking with PRISM (minmax)
        command = f"/Users/thobad/Documents/Tools/prism/prism/bin/prism {filename} -pf 'Rminmax=? [ F \"goal\" ];' -exportstrat 'prism/policy_minmax.txt' -exportvector 'prism/values_minmax.txt'"
        subprocess.Popen(command, shell=True).wait() 

        time_prism = time.time() - time0

        grid.read_prism_output(values='prism/values_minmax.txt', policy='prism/policy_minmax.txt')
        fraction_minmax = grid.fraction_besteffort

        # Remove non-optimal actions from the model
        grid.remove_nonoptimal_actions()
        # Write the updated model to a new PRISM file
        filename = 'prism/slipgrid_{}_{}_seed={}_minmin.nm'.format(X, Y, seed)
        grid.write_to_prism(filename)

        # Model checking with PRISM (minmin)
        command = f"/Users/thobad/Documents/Tools/prism/prism/bin/prism {filename} -pf 'Rminmin=? [ F \"goal\" ];' -exportstrat 'prism/policy_minmin.txt' -exportvector 'prism/values_minmin.txt'"
        subprocess.Popen(command, shell=True).wait()

        time_prism_double = time.time() - time0 - time_prism

        grid.read_prism_output(values='prism/values_minmin.txt', policy='prism/policy_minmin.txt')
        fraction_minmin = grid.fraction_besteffort

        print(f'\nFraction of best-effort actions (minmax): {fraction_minmax:.2f}')
        print(f'Fraction of best-effort actions (minmin): {fraction_minmin:.2f}')

    if storm:
        grid.write_to_storm()

    return fraction_minmax, fraction_minmin, time_prism, time_prism_double

storm = False
prism = True

Xbase = 10     # X cells of the grid
Ybase = 10     # Y cells of the grid
Bbase = 10     # Number of obstacles
sizes = [1,5,10]
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
        be_minmax = np.zeros(len(seeds))
        be_minmin = np.zeros(len(seeds))
        time_prism = np.zeros(len(seeds))
        time_prism_double = np.zeros(len(seeds))
        
        # Run the instances for each random seed
        for i,seed in enumerate(seeds):

            # Run the instance with the given parameters
            be_minmax[i], be_minmin[i], time_prism[i], time_prism_double[i] = run_instance(X, Y, B, P=P, seed=seed, storm=storm, prism=prism, verbose=False)

        # Store the results for this probability P
        results[(size, P)]['prism-time'] = f'${np.round(np.mean(time_prism), 1)} \pm {np.round(np.std(time_prism), 1)}$'
        results[(size, P)]['prism-frac'] = f'${np.round(np.mean(be_minmax), 1)} \pm {np.round(np.std(be_minmax), 1)}$'
        results[(size, P)]['ours-time'] = f'${np.round(np.mean(time_prism_double), 1)} \pm {np.round(np.std(time_prism_double), 1)}$'
        results[(size, P)]['ours-frac'] = f'${np.round(np.mean(be_minmin), 1)} \pm {np.round(np.std(be_minmin), 1)}$'

DF = pd.DataFrame.from_dict(results, orient='index')
print(DF)
DF.to_csv('results/slipgrid_results.csv')
with open('results/slipgrid_results.tex', 'w') as tf:
     tf.write(DF.to_latex())

# Generate IMDP with Storm
# nr_states, obstacles = define_slipgrids(X=X, Y=Y, B=B, seed=0, p_slip_min=0.1, p_slip_max=0.25, p_to_sink=0.1, threshold = P, verbose=True)

# Run PRISM command
# /Users/thobad/Documents/Tools/prism/prism/bin/prism prism/slipgrid_5_5_seed=0.nm -pf 'Rminmax=? [ F "goal" ];' -exportstrat 'prism/policy.txt' -exportvector 'prism/values.txt'