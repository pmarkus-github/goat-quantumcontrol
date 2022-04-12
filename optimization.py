import numpy as np
from scipy.integrate import solve_ivp
from scipy import optimize
import qutip as qt

class Optimizer:


    def __init__(self):
        self.H0 = None
        self.Hdrive = None
        self.initial = None
        self.target = None
        self.evolvedState = None
        self.pulse = None
        self.max_iter = None
        self.xtol = None
        self.ftol = None
        self.infidelity = None
        self.dimensions = None
        self.iterations = 0

    def run_optimization(self, H0, Hdrive,
                         initial, target,
                         pulse, dimensions,
                         max_iter,
                         xtol, ftol):

        self.H0 = H0
        self.Hdrive = Hdrive
        self.initial = initial
        self.target = target
        self.pulse = pulse
        self.dimensions = dimensions
        #self.pulse.get_fourier_pulse()
        guess_amps = self.pulse.guess_amps

        result = optimize.fmin_l_bfgs_b(func=self.compute_infidelity, x0=guess_amps,
                                        disp=1, factr=10, pgtol=1e-20, maxls=70,
                                        approx_grad=False)

        return result

    def compute_infidelity(self, amps):

        self.pulse.amps = amps[:-2]
        self.pulse.w = amps[-2]
        self.pulse.phase = amps[-1]

        result = self.compute_gradient_and_evolution()

        #psi_initial = self.initial
        #psi_target = self.target
        U_initial = self.initial
        U_target = self.target
        # define psu_target dagger (conjugate of transpose)
        #psi_target_dag = np.conjugate(psi_target.T)
        U_target_dag = U_target.T

        #psi_evolved = result[0]@psi_initial
        U_evolved = result[0]
        dU = result[1]
        index = 0
        d = self.dimensions
        d_2 = d ** 2
        gradients = np.empty(self.pulse.total_params)
        g = np.trace(U_target_dag @ U_evolved)

        for k in range(self.pulse.total_params):
            dU_k = dU[index:index + d_2].reshape((d, d))
            #grad_calc = -2 * np.real(np.abs(U_target_dag @ dU_k @ U_initial))
            gradient = - 0.5 * np.real(np.trace(U_target_dag @ dU_k))
            gradients[k] = gradient
            index += d_2

        #print('Norm of state: {}' .format(np.linalg.norm(U_evolved @ U_initial)))
        infidelity = 1 - 0.5 * np.abs(g)
        print('Infidelity: {}' .format(infidelity))
        self.infidelity = infidelity
        self.iterations += 1

        return infidelity, gradients



    def compute_gradient_and_evolution(self):
        num_total_params = self.pulse.total_params
        t = self.pulse.t
        d = self.dimensions
        d_2 = d**2

        A0 = np.zeros(((num_total_params + 1)*d_2, ))
        A0[0:d_2] = np.identity(self.dimensions).reshape((d_2, ))
        A0 = A0.astype(complex)

        f = lambda t, A: self.evolution(t, A)

        res = solve_ivp(f, (t[0], t[-1]), A0.flatten(),
                        t_eval=t, rtol=1e-10, atol=1e-10)

        UT = res.y[0:d_2, -1].reshape((d, d))
        gradients = res.y[d_2:, -1]

        return UT, gradients

    def evolution(self, t, A):

        A = np.reshape(A, (-1, ))
        d = self.dimensions
        d_2 = d**2
        num_total_params = self.pulse.total_params
        num_fourier_params = self.pulse.num_fourier_params
        amps = self.pulse.amps
        w = self.pulse.w
        phase = self.pulse.phase
        evo_time = self.pulse.evo_time
        sin_contribution = 0

        for j in range(1, num_fourier_params + 1):
            sin_contribution += amps[j-1] * np.sin(2 * np.pi * j * t/evo_time)

        pulse = np.cos(w * t + phase) * sin_contribution

        H = self.H0 + pulse * self.Hdrive

        H_kron_id = np.kron(np.identity(d), H)

        Hdrive_kron_id = np.kron(np.identity(d), self.Hdrive)

        dAdt = np.zeros(((num_total_params + 1)*d_2, )).astype(complex)
        dAdt[0:d_2] = -1j*H_kron_id@A[0:d_2]

        l_index = d_2

        for i in range(1, num_fourier_params+1):
            ubound_l_index = l_index+d_2

            dAdt[l_index:ubound_l_index] = -1j * (np.cos(w * t + phase) * np.sin(2 * np.pi * i * t/evo_time) *
                                                  Hdrive_kron_id @ A[0:d_2] +
                                                  H_kron_id @ A[l_index:ubound_l_index])

            l_index += d_2

        temp_w = -1 * t * np.sin(w * t + phase) * sin_contribution * Hdrive_kron_id
        temp_phase = -1 * np.sin(w * t + phase) * sin_contribution * Hdrive_kron_id

        dAdt[l_index:l_index + d_2] = -1j * (temp_w @ A[0:d_2] + H_kron_id @ A[l_index:l_index + d_2])
        l_index += d_2
        dAdt[l_index:l_index + d_2] = -1j * (temp_phase @ A[0:d_2] + H_kron_id @ A[l_index:l_index + d_2])

        return dAdt





