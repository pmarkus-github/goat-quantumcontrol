import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy import optimize


class Optimizer:

    def __init__(self, H0, Hdrive,
                 initial, target,
                 pulse, dimensions,
                 max_iter, ftol,
                 printProgress=None):

        self.H0 = H0
        self.Hdrive = Hdrive
        self.H0_kron_id = None
        self.Hdrive_kron_id = None
        self.dAdt = None
        self.A0 = None
        self.initial = initial
        self.target = target
        self.evolved = None
        self.pulse = pulse
        self.max_iter = max_iter
        self.xtol = None
        self.ftol = ftol
        self.infidelity = None
        self.dimensions = dimensions
        self.dimensions_sq = dimensions**2
        self.options = None
        self.printProgress = printProgress
        self.callBack = None
        self.result = None
        self.iterations = None


    def run_optimization(self):

        self.iterations = 0

        d_2 = self.dimensions_sq
        self.H0_kron_id = np.kron(self.H0, np.identity(self.dimensions))
        self.Hdrive_kron_id = np.kron(self.Hdrive, np.identity(self.dimensions))

        self.A0 = np.zeros(((self.pulse.total_params + 1) * d_2,))
        self.A0[0:d_2] = np.identity(self.dimensions).reshape((d_2,))
        self.A0 = self.A0.astype(complex)

        self.dAdt = np.zeros(((self.pulse.total_params + 1)*d_2, )).astype(complex)
        self.options = {
            'maxiter': self.max_iter,
            'disp': True
        }
        guess_amps = self.pulse.guess_amps

        if self.printProgress is not None and self.printProgress is True:
            self.callBack = self.print_progress_per_iter

        self.result = optimize.minimize(fun=self.compute_infidelity, x0=guess_amps,
                                        method='BFGS', jac=True,
                                        options=self.options,
                                        callback=self.callBack)

        return self


    def compute_infidelity(self, amps):

        self.pulse.model_params = amps

        result = self.compute_gradient_and_evolution()

        U_target = self.target
        # define U_target dagger (conjugate of transpose)
        U_target_dag = np.conjugate(U_target.T)

        # declare the evolved unitary and then only take the relevant 2x2 part of the top left corner
        # (because we are only interested in the two-dimensional subspace)
        U_evolved = result[0]
        U_evolved_sub2 = U_evolved[0:2, 0:2]
        # declare the derivatives of the evolved unitary
        dU = result[1]

        index = 0
        d = self.dimensions
        d_2 = d ** 2
        gradients = np.empty(self.pulse.total_params)

        g = np.trace(U_target_dag @ U_evolved_sub2)
        infidelity = 1 - 0.5 * np.abs(g)

        factor = 0.5 * np.conjugate(infidelity)/np.abs(infidelity)

        for k in range(self.pulse.total_params):
            dU_k = dU[index:index + d_2].reshape((d, d))
            dU_k = dU_k[0:2, 0:2]
            gradient = - np.real(factor * np.trace(U_target_dag @ dU_k))
            gradients[k] = gradient
            index += d_2

        self.infidelity = infidelity
        self.evolved = U_evolved

        return infidelity, gradients


    def compute_gradient_and_evolution(self):
        t = self.pulse.t
        d = self.dimensions
        d_2 = self.dimensions_sq

        A0 = self.A0

        evo = lambda x, A: self.evolution(x, A)

        res = solve_ivp(evo, (t[0], t[-1]), A0.flatten(),
                        t_eval=t, rtol=1e-10, atol=1e-10)

        UT = res.y[0:d_2, -1].reshape((d, d))
        gradients = res.y[d_2:, -1]

        return UT, gradients


    def evolution(self, t, A):

        d_2 = self.dimensions_sq
        pulse, derivatives = self.pulse.evalPulseAndDerivatives(t)

        H_kron_id = self.H0_kron_id + pulse * self.Hdrive_kron_id
        Hdrive_kron_id = self.Hdrive_kron_id

        dAdt = self.dAdt
        dAdt[0:d_2] = -1j * H_kron_id @ A[0:d_2]

        l_index = d_2

        for derivative in derivatives:
            ubound_l_index = l_index + d_2

            dAdt[l_index:ubound_l_index] = -1j * (derivative * Hdrive_kron_id @ A[0:d_2] +
                                           H_kron_id @ A[l_index:ubound_l_index])

            l_index += d_2

        return dAdt


    def print_progress_per_iter(self, data):

        print('Iteration: {}; Infidelity: {}' .format(self.iterations, self.infidelity))
        self.iterations += 1
