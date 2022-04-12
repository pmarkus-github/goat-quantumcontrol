import numpy as np


class FourierPulse:

    def __init__(self, t, evo_time, num_fourier_amps):
        self.num_ctrls = None
        self.num_fourier_params = num_fourier_amps
        self.total_params = None
        self.amps = None
        self.w = None
        self.phase = None
        self.t = t
        self.n_ts = t.size
        self.evo_time = evo_time
        self.guess_amps = None
        self.pulse = None
        #self.phase = self.define_phases()
        #self.gaussian_window = self.get_window_function_gaussian()

    def create_guess_amps(self):

        #amps = (1 * np.random.rand(self.num_fourier_params) - 0.5)
        amps = (1 * np.random.rand(self.num_fourier_params) - 0.5)

        w = 0.1
        phase = 0.1
        self.w = w
        self.phase = phase
        amps = np.append(amps, [w])
        amps = np.append(amps, [phase])

        self.total_params = self.num_fourier_params + 2
        self.guess_amps = amps
        self.amps = amps

        # amp_sin = (2 * np.random.rand(self.num_params) - 2)
        # amp_cos = (2 * np.random.rand(self.num_params) - 2)
        #
        # self.guess_amps = np.column_stack((amp_sin, amp_cos))
        # self.amps = self.guess_amps

    def define_phases(self):

        freq_coeff = np.arange(1, self.num_params + 1)
        factor = (2*np.pi*self.t/self.evo_time)

        phases = factor.reshape((-1, 1))@freq_coeff.reshape((1, -1))

        return phases

    def get_fourier_pulse(self):

        phases = self.phase
        amps = self.amps
        amps = amps.reshape((self.num_params, 2))
        sin_amps = amps[:, 0]
        cos_amps = amps[:, 1]

        pulse = 0
        for k in range(self.num_params):
            pulse += sin_amps[k]*np.sin(phases[:, k]) + cos_amps[k] * np.cos(phases[:, k])

        self.pulse = pulse*self.gaussian_window

    def get_fourier_pulse_grad_sin(self, k, t):

        return np.sin(2*np.pi*k*t/self.evo_time)

    def get_fourier_pulse_grad_cos(self, k, t):

        return np.cos(2*np.pi*k*t/self.evo_time)

    def get_window_function_gaussian(self):

        T = self.evo_time
        td = T / 10

        pulse = (1 - np.exp(-(self.t - T)**2/td) - np.exp(-self.t**2/td))

        return pulse


