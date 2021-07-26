# main_cosine.py - main script for testing cosine function
#
#	created: 7/21/2021
#	last change: 7/21/2021

from network import *
from func import *

import numpy as np
import matplotlib.pyplot as plt

# network parameters
N_neurons = 150
N_outdims = 1
gg = 1.5
gz = 1
p_gg = 0.1
p_gz = 1
p_w = 1
alpha = 1
tau = 10
sigma_gg = np.sqrt(1/(p_gg*N_neurons))
sigma_w = 1

# training parameters
epochs = 5
training_dur = 5000
dt = 1

# plotting parameters
snapshot_len = 2000
plot_int = 10

# function parameters
target_amp = 1.5
target_freq = 10 # Hz


f = Cosine(target_amp, target_freq)

RNN = Network(N_neurons, N_outdims, gg, gz, alpha, sigma_gg, sigma_w, p_gg, p_gz, p_w, tau)

# plot prior to training
x = np.zeros((snapshot_len), dtype=np.float32)
for t in range(0, 10):
	x[t] = RNN.step(f(t-1), dt)

for t in range(10, snapshot_len):
	x[t] = RNN.step(x[t-1], dt)

plt.plot(x)
plt.show()

#train
training(N=RNN, epochs=epochs, training_dur=training_dur, snapshot_len=snapshot_len, \
			plot_int=plot_int, dt=dt, f=f)

# plot after training
# plot function f
x = np.zeros((snapshot_len), dtype=np.float32)
for t in range(snapshot_len):
	x[t] = f(t)

plt.plot(x)

# plot network output
y = np.zeros((snapshot_len), dtype=np.float32)
for t in range(snapshot_len):
	y[t] = RNN.step(y[t-1], dt)

plt.plot(y)
plt.show()