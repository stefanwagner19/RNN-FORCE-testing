from snn import S_RNN

import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt


class IntMatchInput(object):
	def __init__(self, spiketime, dur, dt):
		self.spiketime = spiketime
		self.dur = dur

	def __call__(self, x):
		if x < self.dur:
			return 1
		elif x < self.spiketime or x >= self.spiketime + self.dur:
			return 0
		else:
			return 1

def create_beta(a, b, spike, end, dt):
	end = round(end/dt)
	spike = round(spike/dt)
	x = np.zeros((end, 1, 1))
	
	y = np.linspace(beta.ppf(0, a, b), beta.ppf(1, a, b), round(0.5/dt))
	x[spike-round(0.25/dt):spike+round(0.25/dt), 0, 0] = beta.pdf(y, a, b)

	return x


class Hint(object):
	def __init__(self, spiketime):
		self.spike = spiketime
		self.dx = 1

	def __call__(self, x):
		if x <= self.spike:
			return self.dx*x
		if x <= 2*self.spike:
			return self.dx*self.spike-self.dx*(x-self.spike)
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


def input_spike(t):
	if t <= 0.05:
		return 0.2
	else:
		return 0


def training(Network, f, f_out, h, trials, snapshot_len, plot_int, dur, p):

	print("<--- STARTING TRAINING --->\n")

	Network.Gen.reset_activity()
	Network.Per.reset_activity()

	for i in range(1):
		for t in range(dur):

			Network.Gen.step(f, t, f_out=f_out, h=h)
			Network.Per.step(f, t)


	for trial in range(trials):
		print(f"- Trial {trial+1} -")
		Network.train_once(dur, f, f_out, h, p)
		print(f"--- Avg Spike-Rate: {np.mean(Network.Per.spike_count)/(dur*Network.dt)} Hz")

		# if trial%plot_int == 0 or trial == trials-1:

		# 	Network.Gen.w = Network.Per.w

		# 	x = np.zeros(snapshot_len)
		# 	for t in range(0, snapshot_len):
		# 		x[t] = Network.step(f, t)

		# 	y = np.zeros(snapshot_len)
		# 	for t in range(0, snapshot_len):
		# 		y[t] = f_out[t]

		# 	z = np.zeros(snapshot_len)
		# 	Network.Gen.w = Network.Per.w.copy()
		# 	for t in range(0, snapshot_len):
		# 		z[t] = Network.Gen.step(f, t, f_out, h)

		# 	plt.plot(x, label="output")
		# 	plt.plot(y, label="target")
		# 	plt.plot(z, label="Gen")
		# 	plt.legend(loc="upper left")
		# 	plt.show()


def test(Network, f, f_out, h, init_trials, snapshot_len, dur):
	Network.Gen.reset_activity()
	Network.Per.reset_activity()

	for i in range(init_trials):
		for t in range(dur):
			Network.Gen.step(f, t, f_out=f_out, h=h)
			Network.Per.step(f, t)

	x = np.zeros(snapshot_len)
	y = np.zeros(snapshot_len)
	
	total_error = 0
	
	Network.reset_spike_count()

	for t in range(0, snapshot_len):
		x[t] = Network.step(f, t)
		y[t] = f_out[t]
		total_error += (x[t]-y[t])**2 

	print(f"-- MSE for test run : {total_error/snapshot_len} --")
	print(f"-- Avg spike-Rate: {np.mean(Network.Per.spike_count)/(dur*Network.dt)} Hz")

	plt.plot(x, label="output")
	plt.plot(y, label="target")
	plt.legend(loc="upper left")
	plt.show()
