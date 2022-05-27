import numpy as np
from scipy.linalg import eigh_tridiagonal


def solve_1d_system(potential_1d, dimensions, num_of_intervals, from_pos, to_pos):
    # define the function to calculate the matrix-elements (i,j) of the transition dipole element for our system
    def n_phi_m(My, v, dy):
        # fill the transition-dipole-element with its matrix-elements
        m = np.zeros((dimensions, dimensions))
        for i in range(0, dimensions):
            for j in range(0, dimensions):
                m[i][j] = np.dot(v.T[i], np.matmul(My, v.T[j]))
        return m

    # Interval size (=Total_size/Number_of_steps)
    dy = 4 * np.pi / num_of_intervals
    # Create an evenly spaced array of some total-length with size N
    y = np.linspace(from_pos, to_pos, num_of_intervals + 1)

    My = y[1:-1] * np.identity(num_of_intervals - 1)

    V = potential_1d(y)

    # define the diagonal elements of our tri-diagonal matrix (more of it in the notes)
    d = 2 / (dy ** 2) + V[1:-1]
    # define the two off-diagonal elements of our matrix
    e = -1 / (dy ** 2) * np.ones(len(d) - 1)

    w, v = eigh_tridiagonal(d, e)

    H0 = w[0:dimensions] * np.identity(dimensions)
    # fill the transition-dipole-element with its according matrix-elements

    Hdrive = n_phi_m(My, v, dy)

    return [H0, Hdrive]
