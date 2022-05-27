import numpy as np
import pulses
from optimization import Optimizer
from system_solver import solve_1d_system
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.fftpack import fft


def main():
    def is_unitary(m):
        return np.allclose(np.eye(m.shape[0]), np.conjugate(m.T) @ m)

    def seedGate(alpha, dimensions):

        gate = np.identity(dimensions) * np.cos(alpha / 2) - 1j * sy * np.sin(alpha / 2)

        return gate

    def seedPulseToGate(seed_optimizer, pulse, alpha, stepSize, num_of_iters, dimensions):

        seeded_amps = np.zeros((num_of_iters, 2), dtype=object)
        counter = 0
        for i in range(num_of_iters):
            gate = seedGate(alpha, dimensions)
            seed_optimizer.target = gate
            seed_optim = seed_optimizer.run_optimization()

            result = seed_optim.result

            print("Epsilon: {}; {}; Final infidelity: {}".format(alpha, result.message, result.fun))
            amps_from_seeding = result.x
            seed_optimizer.pulse.guess_amps = amps_from_seeding
            seeded_amps[counter, :] = np.array([alpha, amps_from_seeding], dtype=object)
            alpha += stepSize
            counter += 1

        return seeded_amps

    def seedEvoTimes(seed_optimizer, evo_time_lBound, stepSize, num_of_iters, psi_super, psi_init):

        infidelities = np.zeros((num_of_iters, 3))
        counter = 0
        constant = 1.1
        # 1.21 seems optimal

        for j in range(num_of_iters):
            evolution_time = constant * evo_time_lBound
            seed_optimizer.pulse.evo_time = evolution_time
            seed_optim = seed_optimizer.run_optimization()
            result = seed_optim.result

            gate = seed_optim.evolved


            infidelities[counter, :] = np.array([constant, evolution_time, result.fun])
            print("Evo_Time: {}; {}; Final infidelity: {}".format(evolution_time, result.message, result.fun))
            plt.scatter(counter, result.fun)
            plt.pause(0.05)
            counter += 1
            constant += stepSize

        plt.show()

        return infidelities

    def potential_1d(y, lambda_=0.2, phi_ratio=0.5):

        # lambda_ is defined as the dimensionless quantity 0.5*E_L/E_J (more in the notes)
        # define the ratio of E_J/E_C
        ej_ec = 14

        potential = ej_ec / 4 * (lambda_ * y ** 2 - np.cos(y + phi_ratio * 2 * np.pi))
        min_value = np.min(potential)
        return potential - min_value


    # Dimension of system
    dimension = 2

    # Number of evenly spaced intervals for the finite difference method
    N = 3000
    # define the interval length over which the system should be solved
    interval_from = -2 * np.pi
    interval_to = 2 * np.pi

    hamiltonians = solve_1d_system(potential_1d,
                                   dimensions=8,
                                   num_of_intervals=N,
                                   from_pos=interval_from,
                                   to_pos=interval_to)

    sz = np.array([[1, 0],
                   [0, -1]])

    sx = np.array([[0, 1],
                   [1, 0]])

    sy = np.array([[0, -1j],
                   [1j, 0]])

    not_gate_4d = np.array([[0, 1, 0, 0],
                            [1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

    not_gate_2d = np.array([[0, 1],
                            [1, 0]])

    hardamad_1 = 1 / np.sqrt(2) * np.array([[1, 1],
                                            [1, -1]])

    H0 = hamiltonians[0]
    Hdrive = hamiltonians[1]

    w10 = H0[1, 1] - H0[0, 0]
    non_linearity_relative = (H0[2, 2] - H0[1, 1]) / (H0[1, 1] - H0[0, 0]) - 1
    non_linearity = non_linearity_relative * w10
    evo_time_lbound = 2 * np.pi / np.abs(non_linearity)

    H0 = H0[0:dimension, 0:dimension]
    Hdrive = Hdrive[0:dimension, 0:dimension]
    epsilon = 0

    # define the finale and initial state
    psi_initial = np.eye(1, dimension, 0).reshape((dimension, 1))
    psi_target = np.eye(1, dimension, 1).reshape((dimension, 1))
    psi_superposition = 1 / np.sqrt(2) * (psi_initial + psi_target)

    U_initial = np.identity(dimension)
    R_Ypi4 = np.identity(2) * np.cos(np.pi / 8) - 1j * sy * np.sin(np.pi / 8)

    # define the number of time intervals
    n_ti = 10000
    # define the evolution time
    evo_time = evo_time_lbound
    # create an evenly spaced nd array between 0 and evo_time
    t = np.linspace(0, evo_time, n_ti)
    # define the number of params to be optimized for one basis function
    num_fourier_params = 10

    # create an instance of the FourierPulse class
    fourier_pulse = pulses.FourierPulseWithEnvelope(t=t,
                                                    evo_time=evo_time,
                                                    num_fourier_amps=num_fourier_params)

    # create initial guess_amps
    fourier_pulse.create_guess_amps(w10)

    #initial_f = fourier_pulse.get_pulse(fourier_pulse.guess_amps)
    U_target = R_Ypi4
    print(is_unitary(U_target))
    # run the optimization
    optimizer = Optimizer(H0=H0, Hdrive=Hdrive,
                          initial=U_initial, target=U_target,
                          pulse=fourier_pulse, dimensions=dimension,
                          max_iter=200, ftol=1e-8,
                          printProgress=True)
    step_size = 0.05
    # infids = seedEvoTimes(seed_optimizer=optimizer, evo_time_lBound=evo_time,
    #                       stepSize=step_size, num_of_iters=8,
    #                       psi_super=psi_superposition, psi_init=psi_initial)


    evolution_time = 1.21 * evo_time_lbound
    fourier_pulse.evo_time = evolution_time
    # step_size = np.pi/20
    # amps = seedPulseToGate(optimizer, fourier_pulse, 0, step_size, 20, 2)

    #print(infids)
    #evo_time *= 1.21

    fourier_pulse.evo_time = evolution_time
    optimizer.run_optimization()
    fourier_pulse.plot_pulse(optimizer.result.x)

    # Number of samplepoints
    # Number of samplepoints
    N = 800
    # sample spacing
    T = evo_time / 800.0
    x = np.linspace(0.0, N * T, N)
    fourier_pulse.t = x
    y = fourier_pulse.get_pulse(optimizer.result.x)
    yf = fft(y)
    xf = np.linspace(0.0, evo_time / (1.0 * T), N // 2)

    fig, ax = plt.subplots()
    ax.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))
    plt.show()
    hamiltonians = solve_1d_system(potential_1d,
                                   dimensions=15,
                                   num_of_intervals=N,
                                   from_pos=interval_from,
                                   to_pos=interval_to)

    def gateEvo(t, A, dimension, H0, Hd, pulse):

        dAdt = np.zeros((dimension**2, )).astype(complex)
        params = pulse.amps
        amp = pulse.amplitude
        phase = pulse.phase
        num_params = pulse.num_fourier_params
        evo_time = pulse.evo_time
        w = pulse.w
        pulse = 0
        for j in range(1, num_params + 1):
            pulse += params[j - 1] * np.sin(2 * np.pi * j * t / evo_time)

        pulse *= np.cos(w * t + phase)
        pulse *= amp

        H = H0 + pulse * Hd
        H_kron_id = np.kron(np.identity(counter), H)

        dAdt[0:counter**2] = -1j * H_kron_id @ A[0:counter**2]

        return dAdt

    counter = 2
    infidelities = np.zeros((13, 2))
    for i in range(13):
        A0 = np.zeros((counter**2,))
        A0[0:counter**2] = np.identity(counter).reshape((counter**2,))
        A0 = A0.astype(complex)
        H0 = hamiltonians[0][0:counter, 0:counter]
        Hd = hamiltonians[1][0:counter, 0:counter]

        f = lambda t, A: gateEvo(t, A, counter, H0, Hd, fourier_pulse)

        res = solve_ivp(f, (fourier_pulse.t[0], fourier_pulse.t[-1]), A0.flatten(),
                        t_eval=fourier_pulse.t, rtol=1e-10, atol=1e-10)

        gate = res.y[0:counter**2, -1].reshape((counter, counter))
        U_target_ = np.identity(counter).astype(complex)
        U_target_[0:2, 0:2] = U_target
        infid = 1 - 1/counter * np.abs(np.trace(np.conjugate(U_target_.T) @ gate))
        infidelities[i, :] = np.array([counter, infid])
        counter += 1


    #print(infids)

    # final_amps_from_normal = optim.result.x
    # fourier_pulse.plot_pulse(final_amps_from_normal)
    # fourier_pulse.plot_pulses(seeded_amps, final_amps_from_normal)
    #
    # print("Final seeded amps: {}" .format(seeded_amps[-1][1]))
    # print("\n\nNot seeded amps: {}".format(final_amps_from_normal))


if __name__ == '__main__':
    main()
