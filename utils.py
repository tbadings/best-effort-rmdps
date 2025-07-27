import cdd
import numpy as np

def xy_to_index(x, y, X, Y):
    """Convert (x, y) coordinates to a single index."""

    # Grid looks like this:
    # 0 1 2 3 4
    # 5 6 7 8 9
    # ... and so on

    # Ensure x and y are within bounds
    y = min(max(0, y), Y - 1)
    x = min(max(0, x), X - 1)

    return x * Y + y

def index_to_xy(index, Y):
    """Convert a single index back to (x, y) coordinates."""

    # Calculate y coordinate
    y = index % Y

    # Calculate x coordinate
    x = index // Y

    return x, y

def compute_polytope_halfspaces(vertices):
    '''
    Compute the halfspace representation (H-rep) of a polytope.
    '''

    t = np.ones((vertices.shape[0], 1))  # first column is 1 for vertices
    tV = np.hstack([t, vertices])
    mat = cdd.matrix_from_array(tV, rep_type=cdd.RepType.GENERATOR) #, number_type="float")
    P = cdd.polyhedron_from_matrix(mat)
    bA = np.array(cdd.copy_inequalities(P).array)

    print(bA)

    # the polyhedron is given by b + A x >= 0 where bA = [b|A]
    b, A = np.array(bA[:, 0]), -np.array(bA[:, 1:])

    return A, b