# main.py - main script for FORCE learning
#
#	created: 7/6/2021
#	last change: 7/16/2021

from params import *
from network import Network

import numpy as np
import matplotlib.pyplot as plt


class Sine():
	def __init__(self, amp, freq):
		self.amp = amp
		self.freq = freq

	def __call__(self, t):
		return self.amp*np.sin(2*np.pi*self.freq*t/1000)


class Cosine():
	def __init__(self, amp, freq):
		self.amp = amp
		self.freq = freq

	def __call__(self, t):
		return self.amp*np.cos(2*np.pi*self.freq*t/1000)


class Sawtooth():
	def __init__(self, amp, freq):
		self.amp = amp
		self.freq = freq

	def __call__(self, t):
		dx = 2*self.amp/(self.freq-1)
		return -self.amp + (t % self.freq) * dx


def training(N, epochs, training_dur, dt, f):
	
	print("--- Starting Training ---")

	for e in range(epochs):

		N.reset_activations()
		N.train(training_dur, dt, f)


epochs = 3
training_dur = 12000
dt = 1

snapshot_len = 2000

target_amp = 1.5
# target_freq = 100 #Hz
target_freq = 10

RNN = Network(N_neurons, N_outdims, gg, gz, alpha, sigma_gg, sigma_w, p_gg, tau)

# f = Sine(target_amp, target_freq)
f = Cosine(target_amp, target_freq)
# f = Sawtooth(target_amp, target_freq)

# plot prior to training
x = np.zeros((snapshot_len), dtype=np.float32)
for t in range(1, snapshot_len):
	x[t] = RNN.step(x[t-1], dt)

plt.plot(x)
plt.show()

#train
training(N=RNN, epochs=epochs, training_dur=training_dur, dt=dt, f=f)

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