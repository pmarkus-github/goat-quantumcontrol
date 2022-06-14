import numpy as np
import time
from . import log
from scipy.integrate import solve_ivp
from scipy import optimize


class Optimizer:

    def __init__(self, H0, Hdrive,
                 target, pulse,
                 max_iter=200, gtol=None,
                 printProgress=False):

        self.H0 = H0
        self.Hdrive = Hdrive
        self.pulse = pulse
        self.max_iter = max_iter
        self.printProgress = printProgress
        self.gtol = gtol
        self.infidelity = None
        self.infidelities = None
        self.options = None
        self.callBack = None
        self.result = None
        self.iterations = None
        self.evolved = None
        self.elapsedTime = None

        # defer the dimensionality of the system from the provided Hamiltonian
        self.dimensions = self.H0.shape[0]
        self.dimensions_sq = self.dimensions**2
        self.sub_dimensions = target.shape[0]
        # compute the kronecker product between H0 and the identity matrix
        self.H0_kron_id = np.kron(self.H0, np.identity(self.dimensions))
        # compute the kronecker product between Hdrive and the identity matrix
        self.Hdrive_kron_id = np.kron(self.Hdrive, np.identity(self.dimensions))


        self.U0 = np.zeros(((self.pulse.total_params + 1) * self.dimensions_sq,))
        self.dUdt = self.U0.copy().astype(complex)

        self.U0[0:self.dimensions_sq] = np.identity(self.dimensions).reshape((self.dimensions_sq,))
        self.U0 = self.U0.astype(complex)

        self.target_dagger = np.conjugate(target.T)

        self.options = {
            'maxiter': self.max_iter,
            'disp': True
        }
        log.init_logger()
        if self.gtol is not None:
            if not isinstance(self.gtol, float):
                log.logger.info('Only float values are accepted for the gtol attribute!')
                exit()
            self.options['gtol'] = self.gtol

        if not isinstance(printProgress, bool):
            log.logger.info('Only boolean values accepted for the printProgress attribute!')
            exit()
        elif printProgress:
            self.callBack = self.print_progress_per_iter


    def run_optimization(self):

        self.iterations = 0
        guess_amps = self.pulse.guess_amps

        if self.printProgress:
            log.logger.info('Initial infidelity:')
            self.compute_infidelity(guess_amps, nargout=1)
            self.print_progress_per_iter(data=None)

        log.logger.info('Start optimization...')
        startTime = time.time()
        self.result = optimize.minimize(fun=self.compute_infidelity, x0=guess_amps,
                                        method='BFGS', jac=True,
                                        options=self.options,
                                        callback=self.callBack)

        endTime = time.time()
        self.elapsedTime = endTime - startTime

        log.logger.info("Elapsed time: {}s" .format(self.elapsedTime))
        return self


    def compute_infidelity(self, amps, nargout=2):

        self.pulse.model_params = amps

        U_target_dag = self.target_dagger

        result = self.compute_gradient_and_evolution(nargout=nargout)

        # declare the evolved unitary and then only take the relevant d_sub x d_sub part of the top left corner
        d_sub = self.sub_dimensions
        # (because we are only interested in the two-dimensional subspace)
        U_evolved = result[0]
        U_evolved_sub2 = U_evolved[0:d_sub, 0:d_sub]

        normalization_factor = 1/d_sub
        g = np.trace(U_target_dag @ U_evolved_sub2)
        infidelity = 1 - normalization_factor * np.abs(g)
        self.infidelity = infidelity
        self.evolved = U_evolved

        if nargout > 1:
            # declare the derivatives of the evolved unitary
            dU = result[1]

            index = 0
            d = self.dimensions
            d_2 = self.dimensions_sq
            gradients = np.empty(self.pulse.total_params)

            factor = normalization_factor * np.conjugate(g)/np.abs(g)

            for k in range(self.pulse.total_params):
                dU_k = dU[index:index + d_2].reshape((d, d))
                dU_k = dU_k[0:d_sub, 0:d_sub]
                gradient = - np.real(factor * np.trace(U_target_dag @ dU_k))
                gradients[k] = gradient
                index += d_2

            return infidelity, gradients

    def compute_gradient_and_evolution(self, nargout):
        t = self.pulse.t
        d = self.dimensions
        d_2 = self.dimensions_sq

        U0 = self.U0

        evo = lambda x, U: self.evolution(x, U)

        res = solve_ivp(evo, (t[0], t[-1]), U0.flatten(),
                        t_eval=t, rtol=1e-10, atol=1e-10)

        UT = res.y[0:d_2, -1].reshape((d, d))
        gradients = res.y[d_2:, -1]
        if nargout > 1:
            return UT, gradients
        else:
            return [UT]

    def evolution(self, t, U):

        d_2 = self.dimensions_sq
        pulse, derivatives = self.pulse.evalPulseAndDerivatives(t)

        H_kron_id = self.H0_kron_id + pulse * self.Hdrive_kron_id
        Hdrive_kron_id = self.Hdrive_kron_id

        dUdt = self.dUdt
        dUdt[0:d_2] = -1j * H_kron_id @ U[0:d_2]

        l_index = d_2

        for derivative in derivatives:
            ubound_l_index = l_index + d_2

            dUdt[l_index:ubound_l_index] = -1j * (derivative * Hdrive_kron_id @ U[0:d_2] +
                                                  H_kron_id @ U[l_index:ubound_l_index])

            l_index += d_2

        return dUdt


    def print_progress_per_iter(self, data):

        log.logger.info('[Iteration: {}]----Infidelity: {}'
                        .format(self.iterations, self.infidelity))

        if self.iterations > 0:
            self.infidelities = np.append(self.infidelities, [self.infidelity])
        else:
            self.infidelities = np.array([self.infidelity])

        self.iterations += 1
