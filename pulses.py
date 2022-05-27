import numpy as np
import scipy.signal.windows
from scipy import signal
from matplotlib import pyplot as plt


def eval_blackman_window(t, evo_time):

    # equal to implementation of scipy
    weights = [0.42, -0.50, 0.08]

    window = 0
    for k in range(len(weights)):
        window += weights[k] * np.cos(k * 2 * np.pi * t/evo_time)

    return window


class Pulse:

    def __init__(self, t, evo_time):
        self.num_ctrls = None
        self.model_params = None
        self.total_params = None
        self.guess_amps = None
        self.pulse = None
        self.evo_time = evo_time
        self.t = t
        self.n_ts = t.size

    def plot_pulse(self, pulse, amps):
        plt.plot(pulse.t, pulse.get_pulse(amps), label='Pulse')

        plt.legend()
        plt.show()


class FourierPulseWithEnvelope(Pulse):

    def __init__(self, t, evo_time, num_fourier_amps):
        super().__init__(t, evo_time)
        self.num_fourier_params = num_fourier_amps
        self.amps = None
        self.w = None
        self.phase = None
        self.amplitude = None

    def create_guess_amps(self, omega10=0.1):

        amps = (1 * np.random.rand(self.num_fourier_params) - 0.5)
        w = omega10
        phase = 0.1
        amplitude = 0.1

        self.amps = amps
        self.w = w
        self.phase = phase
        self.amplitude = amplitude
        self.model_params = np.append(amps, [w, phase, amplitude])

        self.total_params = self.num_fourier_params + 3
        self.guess_amps = self.model_params

    def evalPulseAndDerivatives(self, t):

        self.amps = self.model_params[:-3]
        self.w = self.model_params[-3]
        self.phase = self.model_params[-2]
        self.amplitude = self.model_params[-1]

        derivatives = np.zeros(self.total_params)
        sin_contribution = 0
        for j in range(1, self.num_fourier_params + 1):
            sin_contribution += self.amps[j - 1] * np.sin(2 * np.pi * j * t / self.evo_time)

            # compute the derivative with respect to the fourier coefficients
            temp_derivative = self.amplitude * np.cos(self.w * t + self.phase) * \
                              np.sin(2 * np.pi * j * t / self.evo_time)

            derivatives[j - 1] = temp_derivative

        # compute the derivatives with respect to omega, phase and the amplitude
        temp_w = -1 * self.amplitude * t * np.sin(self.w * t + self.phase) * sin_contribution
        temp_phase = -1 * self.amplitude * np.sin(self.w * t + self.phase) * sin_contribution
        temp_amplitude = np.cos(self.w * t + self.phase) * sin_contribution

        derivatives[-3] = temp_w
        derivatives[-2] = temp_phase
        derivatives[-1] = temp_amplitude

        pulse = self.amplitude * np.cos(self.w * t + self.phase) * sin_contribution

        return pulse, derivatives

    def get_pulse(self, params=None):

        if params is not None:
            amps = params[:-3]
            omega = params[-3]
            phase = params[-2]
            amplitude = params[-1]
        else:
            amps = self.amps
            omega = self.w
            phase = self.phase
            amplitude = self.amplitude

        sin_contribution = 0
        for j in range(1, self.num_fourier_params):
            amp = amps[j - 1]
            sin_contribution += amp * np.sin(2 * np.pi * j * self.t / self.evo_time)

        pulse = amplitude * np.cos(omega * self.t + phase) * sin_contribution

        return pulse

    def plot_pulse(self, amps, **kwargs):

        super().plot_pulse(self, amps)


class FourierSineCosPulse(Pulse):

    def __init__(self, t, evo_time, num_fourier_amps):
        super().__init__(t, evo_time)
        self.num_fourier_params = num_fourier_amps
        self.sin_amps = None
        self.cos_amps = None
        self.window = None

    def create_guess_amps(self, window=True):

        amps_sin = (1 * np.random.rand(self.num_fourier_params) - 0.5)
        amps_cos = (1 * np.random.rand(self.num_fourier_params) - 0.5)

        self.sin_amps = amps_sin
        self.cos_amps = amps_cos
        self.model_params = np.append(amps_sin, amps_cos)

        self.total_params = self.num_fourier_params * 2
        self.guess_amps = self.model_params

        if window: self.window = eval_blackman_window(self.t, self.evo_time)

    def evalPulseAndDerivatives(self, t):

        self.sin_amps = self.model_params[:self.num_fourier_params]
        self.cos_amps = self.model_params[self.num_fourier_params:]

        derivatives = np.zeros(self.total_params)
        pulse = 0
        omega = 2 * np.pi / self.evo_time
        for j in range(1, self.num_fourier_params + 1):
            sin_amp = self.sin_amps[j - 1]
            cos_amp = self.cos_amps[j - 1]
            pulse += sin_amp * np.sin(omega * t)
            pulse += cos_amp * np.cos(omega * t)

            # compute the derivative with respect to the fourier coefficients
            temp_derivative_sin_amps = np.sin(omega * t)
            temp_derivative_cos_amps = np.cos(omega * t)

            derivatives[j - 1] = temp_derivative_sin_amps
            derivatives[self.num_fourier_params + (j - 1)] = temp_derivative_cos_amps

        if self.window is not None:
            window = eval_blackman_window(t, self.evo_time)
            pulse *= window

        return pulse, derivatives

    def get_pulse(self, params=None):

        if params is not None:
            amps_sin = params[:self.num_fourier_params]
            amps_cos = params[self.num_fourier_params:]
        else:
            amps_sin = self.sin_amps
            amps_cos = self.cos_amps

        pulse = 0
        omega = 2 * np.pi / self.evo_time
        for j in range(1, self.num_fourier_params + 1):
            sin_amp = amps_sin[j - 1]
            cos_amp = amps_cos[j - 1]
            pulse += sin_amp * np.sin(omega * self.t)
            pulse += cos_amp * np.cos(omega * self.t)

        if self.window is not None:
            pulse *= self.window

        return pulse

    def plot_pulse(self, amps, **kwargs):

        super().plot_pulse(self, amps)


class FourierSinePulseSeries(Pulse):

    def __init__(self, t, evo_time, num_fourier_amps):
        super().__init__(t, evo_time)
        self.num_fourier_params = num_fourier_amps
        self.amps = None
        self.amplitude = None

    def create_guess_amps(self):

        amps = (1 * np.random.rand(self.num_fourier_params) - 0.5)
        amplitude = 0.1

        self.amps = amps
        self.amplitude = amplitude
        self.model_params = np.append(amps, [amplitude])

        self.total_params = self.num_fourier_params + 1
        self.guess_amps = self.model_params

    def evalPulseAndDerivatives(self, t):

        self.amps = self.model_params[:-1]
        self.amplitude = self.model_params[-1]

        derivatives = np.zeros(self.total_params)
        sin_contribution = 0
        for j in range(1, self.num_fourier_params + 1):
            sin_contribution += self.amps[j - 1] * np.sin(2 * np.pi * j * t / self.evo_time)

            # compute the derivative with respect to the fourier coefficients
            temp_derivative = self.amplitude * np.sin(2 * np.pi * j * t / self.evo_time)

            derivatives[j - 1] = temp_derivative

        # compute the derivatives with respect to the amplitude
        temp_amplitude = sin_contribution

        pulse = self.amplitude * sin_contribution

        derivatives[-1] = temp_amplitude

        return pulse, derivatives

    def get_pulse(self, params=None):

        if params is not None:
            amps = params[:-1]
            amplitude = params[-1]
        else:
            amps = self.amps
            amplitude = self.amplitude

        pulse = 0
        for j in range(1, self.num_fourier_params + 1):
            amp = amps[j - 1]
            pulse += amp * np.sin(2 * np.pi * j * self.t / self.evo_time)

        return pulse * amplitude

    def plot_pulse(self, amps, **kwargs):

        super().plot_pulse(self, amps)
