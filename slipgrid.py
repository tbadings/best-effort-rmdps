import numpy as np
from utils import xy_to_index, index_to_xy

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
        self.s_init = 0

        # Randomly select O unique obstacles indexes (avoid choosing the initial and goal states)
        self.obstacles = np.random.choice(self.nr_states - 2, size=B, replace=False) + 1
        print(f' - Selected {len(self.obstacles)} obstacles: {self.obstacles}')

    def define_IMDP(self):

        self.imdp = {}
        self.rmdp = {}
        self.reward = {}
        self.imdp_pworst = {}
        self.nr_choices = 0

        # Define transition probability function
        for x in range(self.X):
            for y in range(self.Y):
                if self.verbose:
                    print(f'-- Processing state ({x}, {y})')

                # Calculate the state index from (x, y) coordinates
                i = xy_to_index(x, y, self.X, self.Y)

                self.imdp[i] = {}
                self.rmdp[i] = {}
                self.reward[i] = {}

                delta_min = np.round(np.random.uniform(-0.05, 0.05), 3)
                delta_max = np.round(np.random.uniform(-0.05, 0.05), 3)

                pmin = self.p_slip_min + delta_min
                pmax = self.p_slip_max + delta_max
                pmid = (pmin + pmax) / 2

                # IMDP definition
                if x == self.X - 1 and y == self.Y - 1:  # Goal state
                    self.imdp_pworst[i] = [1]
                    
                    # Add probability intervals to the IMDP
                    self.imdp[i]['pass'] = {}
                    self.imdp[i]['pass'][i] = [1.0, 1.0]

                    self.reward[i]['pass'] = 0
                    self.nr_choices += 1

                    if self.verbose:
                        print(f' - State ({x}, {y}) is the goal state, adding pass action.')
                elif i in self.obstacles:
                    self.imdp_pworst[i] = [1]
                    
                    # Add probability intervals to the IMDP
                    self.imdp[i]['pass'] = {}
                    self.imdp[i]['pass'][0] = [1.0, 1.0]

                    self.reward[i]['pass'] = 0
                    self.nr_choices += 1

                    if self.verbose:
                        print(f' - State ({x}, {y}) is an obstacle, adding pass action to sink state.')
                else:
                    self.imdp_pworst[i] = [1 - pmax, pmax]

                    # Define the actions based on the position in the grid
                    if x > 0: # If x > 0, can move left
                        self.nr_choices += 2
                        self.reward[i]['left'] = 1
                        self.reward[i]['left_be'] = 1

                        if np.random.rand() >= self.threshold:     
                            # IMDP                       
                            self.imdp[i]['left'] = {
                                xy_to_index(x - 1, y, self.X, self.Y): [1 - pmax, 1 - pmax],
                                i: [pmax, pmax]
                            }
                            self.imdp[i]['left_be'] = {
                                xy_to_index(x - 1, y, self.X, self.Y): [1 - pmax, 1 - pmin],
                                i: [pmin, pmax]
                            }
                        else:
                            # IMDP
                            self.imdp[i]['left_be'] = {
                                xy_to_index(x - 1, y, self.X, self.Y): [1 - pmax, 1 - pmin],
                                i: [pmin, pmax]
                            }
                            self.imdp[i]['left'] = {
                                xy_to_index(x - 1, y, self.X, self.Y): [1 - pmax, 1 - pmax],
                                i: [pmax, pmax]
                            }

                    if x < self.X - 1: # If x < X-1, can move right
                        self.nr_choices += 2
                        self.reward[i]['right'] = 1
                        self.reward[i]['right_be'] = 1

                        if np.random.rand() >= self.threshold:
                            # IMDP
                            self.imdp[i]['right'] = {
                                xy_to_index(x + 1, y, self.X, self.Y): [1 - pmax, 1 - pmax],
                                i: [pmax, pmax]
                            }
                            self.imdp[i]['right_be'] = {
                                xy_to_index(x + 1, y, self.X, self.Y): [1 - pmax, 1 - pmin],
                                i: [pmin, pmax]
                            }
                        else:
                            # IMDP
                            self.imdp[i]['right_be'] = {
                                xy_to_index(x + 1, y, self.X, self.Y): [1 - pmax, 1 - pmin],
                                i: [pmin, pmax]
                            }
                            self.imdp[i]['right'] = {
                                xy_to_index(x + 1, y, self.X, self.Y): [1 - pmax, 1 - pmax],
                                i: [pmax, pmax]
                            }
                        
                    if y > 0: # If y > 0, can move up
                        self.nr_choices += 2
                        self.reward[i]['up'] = 1
                        self.reward[i]['up_be'] = 1

                        if np.random.rand() >= self.threshold:
                            # IMDP
                            self.imdp[i]['up'] = {
                                xy_to_index(x, y - 1, self.X, self.Y): [1 - pmax, 1 - pmax],
                                i: [pmax, pmax]
                            }
                            self.imdp[i]['up_be'] = {
                                xy_to_index(x, y - 1, self.X, self.Y): [1 - pmax, 1 - pmin],
                                i: [pmin, pmax]
                            }
                        else:
                            # IMDP
                            self.imdp[i]['up_be'] = {
                                xy_to_index(x, y - 1, self.X, self.Y): [1 - pmax, 1 - pmin],
                                i: [pmin, pmax]
                            }
                            self.imdp[i]['up'] = {
                                xy_to_index(x, y - 1, self.X, self.Y): [1 - pmax, 1 - pmax],
                                i: [pmax, pmax]
                            }

                    if y < self.Y - 1: # If y < Y-1, can move down
                        self.nr_choices += 2
                        self.reward[i]['down'] = 1
                        self.reward[i]['down_be'] = 1

                        if np.random.rand() >= self.threshold:
                            # IMDP
                            self.imdp[i]['down'] = {
                                xy_to_index(x, y + 1, self.X, self.Y): [1 - pmax, 1 - pmax],
                                i: [pmax, pmax]
                            }
                            self.imdp[i]['down_be'] = {
                                xy_to_index(x, y + 1, self.X, self.Y): [1 - pmax, 1 - pmin],
                                i: [pmin, pmax]
                            }
                        else:
                            # IMDP
                            self.imdp[i]['down_be'] = {
                                xy_to_index(x, y + 1, self.X, self.Y): [1 - pmax, 1 - pmin],
                                i: [pmin, pmax]
                            }
                            self.imdp[i]['down'] = {
                                xy_to_index(x, y + 1, self.X, self.Y): [1 - pmax, 1 - pmax],
                                i: [pmax, pmax]
                            }

                # RMDP definition                
                if x == self.X - 1 and y == self.Y - 1:  # Goal state

                    # Add probability set to the RMDP
                    self.rmdp[i]['pass'] = {}
                    self.rmdp[i]['pass'][i] = [1.0, 1.0]

                elif i in self.obstacles:

                    # Add probability set to the RMDP
                    self.rmdp[i]['pass'] = {}
                    self.rmdp[i]['pass'][0] = [1.0, 1.0]

                else:
                    # Define the actions based on the position in the grid
                    if x > 0: # If x > 0, can move left
                        if np.random.rand() >= self.threshold:     
                            # RMDP
                            self.rmdp[i]['left'] = {
                                xy_to_index(x - 1, y, self.X, self.Y): [1 - pmax, 1 - pmid],
                                i: [pmax, pmid]
                            }
                            self.rmdp[i]['left_be'] = {
                                xy_to_index(x - 1, y, self.X, self.Y): [1 - pmax, 1 - pmin],
                                i: [pmax, pmin]
                            }
                        else:
                            # RMDP
                            self.rmdp[i]['left_be'] = {
                                xy_to_index(x - 1, y, self.X, self.Y): [1 - pmax, 1 - pmin],
                                i: [pmax, pmin]
                            }
                            self.rmdp[i]['left'] = {
                                xy_to_index(x - 1, y, self.X, self.Y): [1 - pmax, 1 - pmid],
                                i: [pmax, pmid]
                            }

                    if x < self.X - 1: # If x < X-1, can move right
                        if np.random.rand() >= self.threshold:
                            # RMDP
                            self.rmdp[i]['right'] = {
                                xy_to_index(x + 1, y, self.X, self.Y): [1 - pmax, 1 - pmid],
                                i: [pmax, pmid]
                            }
                            self.rmdp[i]['right_be'] = {
                                xy_to_index(x + 1, y, self.X, self.Y): [1 - pmax, 1 - pmin],
                                i: [pmax, pmin]
                            }
                        else:
                            # RMDP
                            self.rmdp[i]['right_be'] = {
                                xy_to_index(x + 1, y, self.X, self.Y): [1 - pmax, 1 - pmin],
                                i: [pmax, pmin]
                            }
                            self.rmdp[i]['right'] = {
                                xy_to_index(x + 1, y, self.X, self.Y): [1 - pmax, 1 - pmid],
                                i: [pmax, pmid]
                            }
                        
                    if y > 0: # If y > 0, can move up
                        if np.random.rand() >= self.threshold:
                            # RMDP
                            self.rmdp[i]['up'] = {
                                xy_to_index(x, y - 1, self.X, self.Y): [1 - pmax, 1 - pmid],
                                i: [pmax, pmid]
                            }
                            self.rmdp[i]['up_be'] = {
                                xy_to_index(x, y - 1, self.X, self.Y): [1 - pmax, 1 - pmin],
                                i: [pmax, pmin]
                            }
                        else:
                            # RMDP
                            self.rmdp[i]['up_be'] = {
                                xy_to_index(x, y - 1, self.X, self.Y): [1 - pmax, 1 - pmin],
                                i: [pmax, pmin]
                            }
                            self.rmdp[i]['up'] = {
                                xy_to_index(x, y - 1, self.X, self.Y): [1 - pmax, 1 - pmid],
                                i: [pmax, pmid]
                            }

                    if y < self.Y - 1: # If y < Y-1, can move down
                        if np.random.rand() >= self.threshold:
                            # RMDP
                            self.rmdp[i]['down'] = {
                                xy_to_index(x, y + 1, self.X, self.Y): [1 - pmax, 1 - pmid],
                                i: [pmax, pmid]
                            }
                            self.rmdp[i]['down_be'] = {
                                xy_to_index(x, y + 1, self.X, self.Y): [1 - pmax, 1 - pmin],
                                i: [pmax, pmin]
                            }
                        else:
                            # RMDP
                            self.rmdp[i]['down_be'] = {
                                xy_to_index(x, y + 1, self.X, self.Y): [1 - pmax, 1 - pmin],
                                i: [pmax, pmin]
                            }
                            self.rmdp[i]['down'] = {
                                xy_to_index(x, y + 1, self.X, self.Y): [1 - pmax, 1 - pmid],
                                i: [pmax, pmid]
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
          f'    x : [0..{(self.X-1)}];',
          f'    y : [0..{(self.Y-1)}];']
    
        for s,v in self.imdp.items():
            for a,w in v.items():
                # Compute the coordinates from the state index
                x,y = index_to_xy(s, self.Y)

                str = [f'[{p[0]}, {p[1]}]: (x\'={index_to_xy(ss, self.Y)[0]}) & (y\'={index_to_xy(ss, self.Y)[1]})' for ss,p in w.items()]
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
            f'label "goal" = x={self.X-1} & y={self.Y-1};'
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
        state_labeling.add_label_to_state('init', xy_to_index(0, 0, self.X, self.Y))
        state_labeling.add_label_to_state('goal', xy_to_index(self.X - 1, self.Y - 1, self.X, self.Y))

        # Add the reward model
        reward_models = {}
        action_reward = [stormpy.pycarl.Interval(r,r) for r in reward_per_choice]
        reward_models["reward"] = stormpy.SparseIntervalRewardModel(optional_state_action_reward_vector=action_reward)

        matrix = builder.build()

        components = stormpy.SparseIntervalModelComponents(transition_matrix=matrix, state_labeling=state_labeling)
        components.choice_labeling = choice_labeling
        imdp = stormpy.storage.SparseIntervalMdp(components)

        if self.verbose:
            print('- IMDP created.')

        prop = stormpy.parse_properties('Rmin=? [F "goal"]')[0]
        env = stormpy.Environment()
        env.solver_environment.minmax_solver_environment.method = stormpy.MinMaxMethod.value_iteration
                   
        if self.verbose:
            print('- Property parsed and environment set up.')

        task = stormpy.CheckTask(prop.raw_formula, only_initial_states=False)
        task.set_produce_schedulers()
        task.set_robust_uncertainty(True)

        if self.verbose:
            print('- Task created.')

        result_generator = stormpy.check_interval_mdp(imdp, task, env)

        if self.verbose:
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

        # Assess which actions are optimal per state
        optimal_actions = {}
        for s,v in self.imdp.items():
            optimal_actions[s] = []
            x,y = index_to_xy(s, self.Y)
            for a,w in v.items():
                value = self.reward[s][a] + np.sum([self.imdp_pworst[s][j] * self.V[ss] for j,ss in enumerate(w.keys())])

                if np.abs(value - self.V[s]) < 1e-4:
                    if self.verbose:
                        print(f' - Value for action {a} in state {s} ({x},{y}) is {value} (optimal)')
                    optimal_actions[s] += [a]
                else:
                    if self.verbose:
                        print(f' - Value for action {a} in state {s} ({x},{y}) is {value} (not optimal)')

        return optimal_actions

    def remove_nonoptimal_actions(self, optimal_actions):
        
        for s,v in self.imdp.items():
            keys = list(v.keys())
            for action in keys:
                if action not in optimal_actions[s]:
                    if self.verbose:
                        print(f' - Removing non-optimal action {action} from state {s}')
                    del self.imdp[s][action]
                    del self.rmdp[s][action]
                    del self.reward[s][action]
                    self.nr_choices -= 1

        # Check if some actions are not enabled in any state anymore
        for action in list(self.rewards.keys()):
            if not any([action in enabled_actions.keys() for enabled_actions in self.imdp.values()]):
                if self.verbose:
                    print(f' - Removing action {action} from rewards as it is not enabled in any state anymore')
                del self.rewards[action]