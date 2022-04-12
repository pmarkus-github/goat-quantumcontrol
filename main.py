import numpy as np
from matplotlib import pyplot as plt
from pulses import FourierPulse
from optimization import Optimizer
from system_solver import solve_1d_sytem


def main():
    def potential_1d(y):

        # b is defined as the dimensionless quantity lambda L/L_j - 1 (more in the notes)
        b = 1
        return (2 * (b + 1) - (b * y ** 2) + (b + 1) * 0.08333 * y ** 4) - 2.49994

    # Dimension of system
    dimension = 2
    # Number of evenly spaced intervals for the finite difference method
    N = 3000
    # define the interval length over which the system should be solved
    interval_from = -2 * np.pi
    interval_to = 2 * np.pi

    hamiltonians = solve_1d_sytem(potential_1d,
                                  dimensions=dimension,
                                  num_of_intervals=N,
                                  from_pos=interval_from,
                                  to_pos=interval_to)

    sz = np.array([[1, 0],
                   [0, -1]])

    sx = np.array([[0, 1],
                   [1, 0]])

    H0 = sz
    Hdrive = sx
    epsilon = 0

    # define the finale and initial state
    #psi_initial = np.array([1, 0, 0, 0]).reshape((dimension, 1)
    # )
    #psi_target = np.array([1, 0, 0, 0]).reshape((dimension, 1))
    U_initial = np.identity(dimension)
    U_target = 1j * np.array([[0, 1],
                              [1, 0]])

    # define the number of time intervals
    n_ti = 1000
    # define the evolution time
    evo_time = 4
    # create an evenly spaced nd array between 0 and evo_time
    t = np.linspace(0, evo_time, n_ti)
    # define the number of params to be optimized for one basis function
    num_fourier_params = 8

    # create an instance of the FourierPulse class
    fourier_pulse = FourierPulse(t=t,
                                 evo_time=evo_time,
                                 num_fourier_amps=num_fourier_params)

    # create initial guess_amps
    fourier_pulse.create_guess_amps()

    initial_f = 0
    for j in range(1, fourier_pulse.num_fourier_params):
        initial_f += fourier_pulse.guess_amps[j-1] * np.sin(2*np.pi * j * fourier_pulse.t/fourier_pulse.evo_time)

    initial_f *= np.cos(fourier_pulse.w * fourier_pulse.t + fourier_pulse.phase)

    # run the optimization
    optimizer = Optimizer()
    result = optimizer.run_optimization(H0=H0, Hdrive=Hdrive,
                                        initial=U_initial, target=U_target,
                                        pulse=fourier_pulse, dimensions=dimension,
                                        max_iter=2000,
                                        xtol=1e-4, ftol=1e-9)

    final_f = 0
    for j in range(1, fourier_pulse.num_fourier_params):
        final_f += result[0][j-1] * np.sin(2*np.pi * j * fourier_pulse.t/fourier_pulse.evo_time)

    final_f *= np.cos(result[0][-2] * fourier_pulse.t + result[0][-1])

    print("Final infidelity: {}" .format(result[1]))
    plt.plot(t, initial_f, label='Initial pulse')
    plt.plot(t, final_f, label='Final pulse')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()
