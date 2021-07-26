# func.py - contains functions essential for training and testing functions
#
#	created: 7/20/2021
#	last change: 7/21/2021


from network import RNN

import numpy as np
import matplotlib.pyplot as plt


class IntMatchFunc(object):
	def __init__(self, spiketime):
		self.spike = spiketime
		self.std = 50

	def __call__(self, x):
		if abs(x - self.spike) < 150:
			return 1/(self.std*np.sqrt(2*np.pi))*np.exp(-(x-self.spike)**2/ \
				(2*self.std**2))*30
		else:
			return 0

class Hint(object):
	def __init__(self, spiketime):
		self.spike = spiketime
		self.peak = 1.2
		self.dx = self.peak/self.spike

	def __call__(self, x):
		if x <= self.spike:
			return self.dx*x
		if x <= 2*self.spike:
			return self.peak-self.dx*x
		else:
			return 0


class Accordian(object):
	def __init__(self, T, upper, lower):
		self.T = T
		self.T_half = T/2
		self.upper = upper
		self.lower = lower

	def __call__(self, x):
		x %= self.T
		dw = (self.upper-self.lower)*np.pi/(self.T_half)
		if x <= self.T_half:
			w = self.lower*np.pi+x*dw
			return np.sin(w*x/(self.T_half))
		if x > self.T_half:
			w = self.upper*np.pi-(x-self.T_half)*dw
			return -np.sin(w*(self.T-x)/self.T_half)

# def Accordian(t):
# 	t %= 400
# 	dw = 4*np.pi/200
# 	if t <= 200:
# 		w = 2*np.pi+t*dw
# 		return np.sin(w*t/200)
# 	if t > 200:
# 		w = 6*np.pi-(t-200)*dw
# 		return -np.sin(w*(400-t)/200)

def input_spike(t):
	if t <= 20:
		return 0.3
	else:
		return 0


def Cosine(t):
	return 1.5*np.cos(2*np.pi*10*t/1000)


def training(Network, f, f_out, h, trials_i, trials_w, snapshot_len, plot_int_i, \
				plot_int_w, dur=1200, dt=1):

	print("<--- STARTING TRAINING --->\n")

	print("internal training ...\n")

	for trial in range(trials_i):
		print(f"internal trial {trial+1}")
		# training for one trial
		Network.internal_training(dur, f, f_out, h, dt)

		# plot internal activity and compare to task-generating network
		if trial%plot_int_i == 0 or trial == trials_i-1:
			
			temp = Network.Per.w
			Network.Per.w = Network.Gen.w
			
			x = np.zeros(snapshot_len)
			for t in range(0, snapshot_len):
				x[t] = Network.step(f, t)

			y = np.zeros(snapshot_len)
			for t in range(0, snapshot_len):
				y[t] = Network.Gen.step(f, t, f_out, h, dt)

			Network.Per.w = temp

			plt.plot(x)
			plt.plot(y)
			plt.show()

	print("output training ...\n")

	for trial in range(trials_w):
		print(f"output trial {trial+1}")
		# training for one trial
		Network.output_training(dur, f, f_out, dt)

		if trial%plot_int_w == 0 or trial == trials_w-1:

			x = np.zeros(snapshot_len)
			for t in range(0, snapshot_len):
				x[t] = Network.step(f, t)

			y = np.zeros(snapshot_len)
			for t in range(0, snapshot_len):
				y[t] = f_out[0](t)

			plt.plot(x)
			plt.plot(y)
			plt.show()


def train2(Network, f, f_out, h, trials, snapshot_len, plot_int, dur, p, dt):

	print("<--- STARTING TRAINING --->\n")

	Network.Gen.reset_activity()
	Network.Per.reset_activity()

	for i in range(3):
		for t in range(dur):

			Network.Gen.step(f, t, f_out=f_out, h=h, dt=dt)
			Network.Per.step(f, t, dt=dt)


	for trial in range(trials):
		print(f"- Trial {trial+1} -")
		Network.train_once(dur, f, f_out, h, p, dt)

		if trial%plot_int == 0 or trial == trials-1:
			x = np.zeros(snapshot_len)
			for t in range(0, snapshot_len):
				x[t] = Network.step(f, t)

			y = np.zeros(snapshot_len)
			for t in range(0, snapshot_len):
				y[t] = f_out[0](t)

			plt.plot(x, label="output")
			plt.plot(y, label="target")
			plt.legend(loc="upper left")
			plt.show()