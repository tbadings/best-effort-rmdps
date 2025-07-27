import numpy as np

g = 0.9

p121 = 1
p122 = 0

denom = (1 - (g**2/2 * p122)/(1-g/2) - g*p121 )**2

xdel1 = (0 - (g*p122)/(1-g/2) * -g) / denom
xdel2 = (g/(1-g/2) * (1 - (g**2/2 * p122)/(1-g/2) - g*p121) - g/(1-g/2) * p122 * -(g**2 / 2)/(1-g/2)) / denom

print(xdel1, xdel2)
print(denom)

print(f'Directional derivative: {-0.2 * xdel1 + 0.2 * xdel2}')