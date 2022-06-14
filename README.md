# goat_quantumcontrol

goat_quantumcontrol is a python library for the optimization of quantum gates using the GOAT algorithm.

The algorithm was developed in 2018 by Machnes et. al. and got published in the following paper:
https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.150401

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install goat_quantumcontrol.

```bash
pip install goat_quantumcontrol
```

## Usage

```python
import goat_quantumcontrol as Qgoat

#-----System parameters------
# define the drift Hamiltonian
H0 = sigmax
# define the control Hamiltonian
Hdrive = sigmaz
# define the target gate
Utarget = X_gate

#-----Pulse parameters------
# define the number of time intervals
n_ts = 1000
# define the evolution time
evo_time = 3
# define the number of amps
num_of_amps = 2

#-----Optimization parameters--
# define the number of maximal iterations
max_iter = 200

# create an instance of the Pulse class to be used
fourier_pulse = Qgoat.FourierPulseWithEnvelope(n_ts=n_ts,
                                               evo_time=evo_time,
                                               num_of_amps=num_of_amps,
                                               window=None)

# create an instance of the Optimizer class
optimizer = Qgoat.Optimizer(H0=H0, Hdrive=Hdrive,
                            target=Utarget,
                            pulse=fourier_pulse,
                            max_iter=max_iter,
                            printProgress=True)

# run the optimization
optimizer.run_optimization()
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)