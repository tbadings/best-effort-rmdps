import numpy as np
import cvxpy as cp

Pa11 = np.array([0, 1, 0])
Pa12 = np.array([0, 0, 1])

Pa21 = np.array([0, 0, 1])
Pa22 = np.array([0, 1, 0])

N = 10000
V = np.zeros((N,3))
V0 = np.array([0,1,0])
V[0] = V0

pi = np.zeros(N+1)
pi0 = np.array([0.2])
pi[0] = pi0

xi = np.zeros(N)

for i in range(N):
    cp_xi = cp.Variable()
    constraints = [
        cp_xi >= 0,
        cp_xi <= 1,
    ]
    objective = (pi[i] * (cp_xi * Pa11 + (1-cp_xi) * Pa12) + (1-pi[i]) * (cp_xi * Pa21 + (1-cp_xi) * Pa22)) @ V[0]
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve()

    # print('- Opt value (inner)', prob.value)
    f = 1
    xi[i] = f * cp_xi.value + (1 - f) * xi[i]
    print(i,'>>> New value for xi is', xi[i])

    del prob, constraints, objective

    cp_pi = cp.Variable()
    constraints = [
        cp_pi >= 0,
        cp_pi <= 1,
    ]
    objective = (cp_pi * (xi[i] * Pa11 + (1-xi[i]) * Pa12) + (1-cp_pi) * (xi[i] * Pa21 + (1-xi[i]) * Pa22)) @ V[0]
    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve()

    # print('- Opt value (outer)', prob.value)
    f = 0.1/np.log(i+2)
    pi[i+1] = f * cp_pi.value + (1-f) * pi[i]
    print(i,'<<< New value for pi is', pi[i+1])

# # Generate a random non-trivial linear program.
# m = 15
# n = 10
# np.random.seed(1)
# s0 = np.random.randn(m)
# lamb0 = np.maximum(-s0, 0)
# s0 = np.maximum(s0, 0)
# x0 = np.random.randn(n)
# A = np.random.randn(m, n)
# b = A @ x0 + s0
# c = -A.T @ lamb0
#
# # Define and solve the CVXPY problem.
# x = cp.Variable(n)
# prob = cp.Problem(cp.Minimize(c.T @ x),
#                   [A @ x <= b])
# prob.solve()
#
# # Print result.
# print("\nThe optimal value is", prob.value)
# print("A solution x is")
# print(x.value)
# print("A dual solution is")
# print(prob.constraints[0].dual_value)