import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from matplotlib.ticker import LinearLocator

g = 0.9
x1 = g/(2-g) / (1-0.5*g+(g/2)/(2-g))
R = np.array([0,1])

size = 11
V = np.zeros((size,size))

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

beta_list = np.linspace(0,1,size)
xi_list = np.linspace(0,0.5,size)

beta_grid, xi_grid = np.meshgrid(beta_list, xi_list)

for i in range(size):
    for j in range(size):
        beta =  beta_grid[i,j]
        xi = xi_grid[i,j]

        P = np.array([
            [beta * (1-xi) + (1-beta) * (1-2*xi),    beta * xi + (1-beta) * 2*xi],
            [0.5,                                    0.5]
        ])

        V[i,j] = (np.linalg.inv(np.eye(2) - g * P) @ R)[0]

# Plot the surface.
surf = ax.plot_surface(beta_grid, xi_grid, V, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

plt.show()

# %%

steps = 11
W = np.zeros(steps)
p_list = np.linspace(0,1,steps)
for i,p in enumerate(p_list):
    W[i] = (1 + 2*g*p + 2*g/(1-g) + (1-p)*(3+g+g/(1-g))/(1-0.5*g)) / (1-g*p/2 - ((1-p)*(g**2/4))/(1-0.5*g))

fig, ax = plt.subplots()
ax.plot(p_list, W)

plt.show()