import numpy as np
import cvxpy as cp
import tqdm

def RVI(rmdp, reward, nr_states, s_init=0, gamma=0.9, iters=100, policy_direction='min', adversary_direction='max', terminate_eps=1e-5):
    """
    Robust Value Iteration algorithm for solving Robust Markov Decision Processes (RMDPs).
    
    This function implements a robust value iteration algorithm that computes optimal policies
    under uncertainty in transition probabilities. The algorithm alternates between optimizing
    the policy for the current uncertainty realization and finding the worst-case uncertainty
    realization for the current policy.
    
    Parameters
    ----------
    rmdp : dict
        Robust MDP representation where rmdp[s][a][s'] contains the uncertainty set
        [p_min, p_max] for transition probability from state s to s' under action a.
    reward : dict
        Reward function where reward[s][a] gives the immediate reward for taking
        action a in state s.
    nr_states : int
        Total number of states in the MDP.
    s_init : int, optional
        Initial state for value reporting (default: 0).
    gamma : float, optional
        Discount factor, must be in [0, 1) (default: 0.9).
    iters : int, optional
        Maximum number of iterations (default: 100).
    policy_direction : str, optional
        Direction for policy optimization, either 'min' or 'max' (default: 'min').
    adversary_direction : str, optional
        Direction for adversary optimization, either 'min' or 'max' (default: 'max').
    terminate_eps : float, optional
        Convergence threshold for value function (default: 1e-5).
    
    Returns
    -------
    Vstar : np.ndarray
        Optimal value function for each state.
    policy : np.ndarray
        Optimal policy (action for each state).
    worstP : np.ndarray
        Worst-case uncertainty realization for each state.
    optimal_actions : dict
        Dictionary mapping each state to a list of all optimal actions in that state.
    
    Notes
    -----
    The algorithm implements a robust value iteration where:
    1. For each state-action pair, the uncertainty set if assumed to be s-rectangular and a convex combination between two vertices.
    2. The algorithm alternates between policy optimization (using current uncertainty
       realization) and adversarial optimization (finding worst-case uncertainty).
    3. Convergence is checked using the L∞ norm of value function differences.
    §
    The uncertainty sets are parameterized by a single variable P[s] ∈ [0,1] such that
    the actual transition probability is P[s] * p_min + (1-P[s]) * p_max.
    
    Examples
    --------
    >>> # Define a simple 2-state RMDP
    >>> rmdp = {0: {'a': {0: [0.1, 0.9], 1: [0.9, 0.1]}},
    ...         1: {'b': {0: [0.0, 0.5], 1: [1.0, 0.5]}}}
    >>> reward = {0: {'a': 1.0}, 1: {'b': 0.0}}
    >>> V, pi, P, opt_actions = RVI(rmdp, reward, nr_states=2)
    """
    
    print('\nStart robust value iteration...')

    # Initialize value function
    V = np.zeros((iters+1, nr_states))
    pi = np.full((iters+1, nr_states), 'none', dtype=object)
    Q = {}
    Qa = {}

    # Initialize point in each uncertainty set (chosen to be the center)
    P = np.zeros((iters+1, nr_states))
    P[0] = 0.5
    
    # Initialize state-action values
    for s,v in rmdp.items():
        Q[s] = np.zeros(len(v))
        Qa[s] = list(v.keys())

    # For each iteration
    z = 0
    tr = tqdm.trange(iters, desc='Current value: 0', leave=True)
    for i in tr:
        tr.set_description(f"Current value in initial state: {V[i,s_init]:.2f}")
        tr.refresh() # to show immediately the update
        z += 1 # Increment iteration counter

        # Optimize for policy under current choice in the uncertainty set
        for s,v in rmdp.items():

            for j,(a,w) in enumerate(v.items()):
                Q[s][j] = reward[s][a] + gamma * np.sum([(P[i,s] * p[0] + (1-P[i,s]) * p[1]) * V[i,ss] for ss,p in w.items()])

            if policy_direction == 'min':
                optimal_action_nr = np.argmin(Q[s])
                V[i+1,s] = np.min(Q[s])
            else:
                optimal_action_nr = np.argmax(Q[s])
                V[i+1,s] = np.max(Q[s])

            pi[i+1,s] = Qa[s][optimal_action_nr]

        # Update point in the uncertainty set
        for s,v in rmdp.items():
            cp_P = cp.Variable()
            constraints = [
                cp_P >= 0,
                cp_P <= 1,
            ]

            w = rmdp[s][pi[i+1,s]]
            objective = reward[s][pi[i+1,s]] + gamma * np.sum([(cp_P * p[0] + (1-cp_P) * p[1]) * V[i,ss] for ss,p in w.items()])

            if adversary_direction == 'min':
                prob = cp.Problem(cp.Minimize(objective), constraints)
            else:
                prob = cp.Problem(cp.Maximize(objective), constraints)

            prob.solve()

            P[i+1,s] = np.round(cp_P.value, 3)

        if np.all(np.abs(V[i+1] - V[i]) <= terminate_eps):
            break

    Vstar = V[z,:]
    policy = pi[z,:]
    worstP = P[z,:]

    # Retrieve set of all optimal actions in each state
    optimal_actions = {}
    for s,v in rmdp.items():
        optimal_actions[s] = []
        for a,w in v.items():
            Q = reward[s][a] + gamma * np.sum([(worstP[s] * p[0] + (1-worstP[s]) * p[1]) * Vstar[ss] for ss,p in w.items()])
            if np.abs(Q - Vstar[s]) < 1e-4:
                optimal_actions[s] += [a]
            
    # Return values from the last one performed before breaking
    return Vstar, policy, worstP, optimal_actions

def compute_derivative(rmdp, reward, V, policy, P, vector, s_init, s_sink, gamma=0.9, maximize=True):
    """
    Compute the directional derivative of the value function V in the direction of the specified vector, under the current policy and transition probabilities.
    """

    # Compute transition probability matrix, such that values can be computed as
    # (I - gamma * P)^{-1} * V = R
    Pfixed = np.zeros((len(V), len(V)))
    Rfixed = np.zeros(len(V))

    # For every state
    for s,v in rmdp.items():
        action = policy[s]
        Rfixed[s] = reward[s][action]
        # For all successor states of this action
        for ss,p in v[action].items():
            # Fill in transition probabilities in the matrix
            Pfixed[s,ss] += (P[s] * p[0] + (1 - P[s]) * p[1])

    Pfull = np.eye(len(V)) - gamma * Pfixed

    # If gamma=1, modify the sink states to ensure P is invertible
    if gamma == 1:
        for s in s_sink:
            Pfull[s,s] = 1

    print('Test, values from inverting:', np.linalg.inv(Pfull) @ Rfixed)

    derivatives = {}
    optimal_actions = {}

    # Now, for every state, compute the derivative by replacing its row with cvxpy parameters
    for s,v in rmdp.items():

        # Compute derivative of value wrt P_param for every action
        action_list = list(v.keys())
        derivatives[s] = {}
        optimal_actions[s] = []

        '''
        # Create decision variables
        cp_V = cp.Variable(len(V))
        constraints = [cp_V >= 0, cp_V <= 1000, cp_V[s_sink] == 0]

        # Add constraints for all states but state s
        mask = np.ones(len(V), dtype=bool)
        mask[s] = False
        mask[s_sink] = False  # Ensure sink state is not included in the constraints
        constraints += [Pfull[mask] @ cp_V == Rfixed[mask]]

        for a,w in v.items():
            P_param = cp.Parameter()
            P_param.value = P[s]  # Set the current value of P for this state

            print(P[s])

            # Add the extra constraint for the current action to complete the equation system
            constraint_current_action = [cp_V[s] >= P_param]

            # Combine all constraints
            objective = cp.Maximize(cp.sum(cp_V)) if maximize else cp.Minimize(cp.sum(cp_V))
            problem = cp.Problem(objective, constraint_current_action)
            print('\nIs dgp?')
            print(problem.is_dgp(dpp=True))

            print(problem)

            # Solve problem
            problem.solve(gp=True, requires_grad=True)
            print('Value:',cp_V[s].value)
            
            # Compute the derivative into a positive change in P_param
            P_param.delta += vector
            problem.derivative()

            derivatives[s][a] = cp_V[s].value + cp_V[s].delta
        '''

        for a,w in v.items():
            rhs = np.zeros(len(V))
            rhs[s] = gamma * np.sum([(1 * p[0] + (-1) * p[1]) * V[ss] for ss,p in w.items()])

            # Derivatives is obtained by inversing the transition matrix and multiplying with the right-hand side
            # vector is the (directional) change in P_param, which is multiplied with the inverse of the transition
            derivatives[s][a] = vector * (np.linalg.inv(Pfull) @ rhs)[s]

        # Compute the actions with the maximum/minimum derivative
        if maximize:
            opt_derivative = np.max(list(derivatives[s].values()))
        else:
            opt_derivative = np.min(list(derivatives[s].values()))

        print(opt_derivative)

        eps = 1e-4
        for a,w in v.items():
            print(f'Action {a} derivative: {derivatives[s][a]}')
            if np.abs(derivatives[s][a] - opt_derivative) < eps:
                optimal_actions[s] += [a]

    return derivatives, optimal_actions