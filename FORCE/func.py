# func.py - contains functions essential for training and testing functions
#
#	created: 7/20/2021
#	last change: 7/21/2021


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
	def __init__(self, amp, T):
		self.amp = amp
		self.T = T

	def __call__(self, t):
		dx = 2*self.amp/(self.T-1)
		return -self.amp + (t % self.T) * dx

def Accordian(t):
	dw = 4*np.pi/1000
	if t <= 1000:
		w = 2*np.pi+t*dw
		return np.sin(w*t/1000)
	if t > 1000:
		w = 6*np.pi-(t-1000)*dw
		return -np.sin(w*(2000-t)/1000)


def training(N, epochs, training_dur, snapshot_len, plot_int, dt, f):
	
	print("--- Starting Training ---")

	for e in range(epochs):

		print(f"Epoch: {e+1}")

		N.reset_activations()
		N.train(training_dur, dt, f)

		if e%plot_int == 0:
			x = np.zeros((snapshot_len), dtype=np.float32)
			for t in range(0, 10):
				x[t] = N.step(f(t-1), dt)

			for t in range(10, snapshot_len):
				x[t] = N.step(x[t-1], dt)

			# plot network output
			y = np.zeros((snapshot_len), dtype=np.float32)
			for t in range(snapshot_len):
				y[t] = f(t)

			plt.plot(x)
			plt.plot(y)
			plt.show()