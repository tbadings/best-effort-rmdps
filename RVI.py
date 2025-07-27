import numpy as np
import cvxpy as cp
import tqdm

def RVI(rmdp, reward, nr_states, s_init=0, gamma=0.9, iters=100, policy_direction='min', adversary_direction='max', terminate_eps=1e-5):
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