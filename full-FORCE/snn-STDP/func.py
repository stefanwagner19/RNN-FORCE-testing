# func.py - contains functions essential for training and testing functions
#
#	created: 8/14/2021
#	last change: 8/14/2021


from snn import S_RNN

import numpy as np
import matplotlib.pyplot as plt


class IntMatchFunc(object):
	def __init__(self, spiketime):
		self.spike = spiketime
		self.std = 60

	def __call__(self, x):
		if abs(x - self.spike) < 300:
			return 225/(self.std*np.sqrt(2*np.pi))*np.exp(-(x-self.spike)**2/ \
				(2*self.std**2))
		else:
			return 0

class IntMatchInput(object):
	def __init__(self, spiketime, dur):
		self.spiketime = spiketime
		self.dur = dur

	def __call__(self, x):
		if x < self.dur:
			return 1
		elif x < self.spiketime or x >= self.spiketime + self.dur:
			return 0
		else:
			return 1


class Hint(object):
	def __init__(self, spiketime):
		self.spike = spiketime
		self.dx = 1/1000

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
	if t <= 50:
		return 2
	else:
		return 0
	# return 2


def Cosine(t):
	return 1.5*np.cos(2*np.pi*10*t/1000)


def training(Network, f, f_out, h, init_trials, trials, snapshot_len, plot_int, dur, p):

	print("<--- STARTING TRAINING --->\n")

	Network.Gen.reset_activity()
	Network.Per.reset_activity()

	for i in range(init_trials):
		Network.Gen.simulate(dur, f, f_out=f_out, h=h)
		Network.Per.simulate(dur, f)
		# for t in range(dur):

		# 	Network.Gen.step(f, t, f_out=f_out, h=h)
		# 	Network.Per.step(f, t)


	for trial in range(trials):
		print(f"- Trial {trial+1} -")
		Network.train_once(dur, f, f_out, h, p)
		print(f"--- Avg Spike-Rate: {np.mean(Network.Per.spike_count)/(dur*Network.dt)*1000} Hz")

		if trial%plot_int == 0 or trial == trials-1:
			# x = np.zeros(snapshot_len)
			# for t in range(0, snapshot_len):
			# 	x[t] = Network.step(f, t)
			x = Network.Per.simulate(dur, f)
			z = Network.Gen.simulate(dur, f, f_out=f_out, h=h)

			y = np.zeros(snapshot_len)
			for t in range(0, snapshot_len):
				y[t] = f_out[t]

			# z = np.zeros(snapshot_len)
			# Network.Gen.w = Network.Per.w.copy()
			# for t in range(0, snapshot_len):
			# 	z[t] = Network.Gen.step(f, t, f_out, h)

			plt.plot(x.flatten(), label="output")
			plt.plot(y, label="target")
			# plt.plot(z, label="Gen")
			plt.legend(loc="upper left")
			plt.show()


def STDP_pre_training(Network, f, f_out, h, init_trials, trials, snapshot_len, plot_int, dur):
	
	print("<--- STARTING PRE-TRAINING --->\n")

	Network.Gen.reset_activity()

	for i in range(init_trials):
		Network.Gen.simulate(dur, f, f_out=f_out, h=h)
		# for t in range(dur):
		# 	Network.Gen.step(f, t, f_out=f_out, h=h)

	for trial in range(trials):
		print(f"- Trial {trial+1}")
		Network.STDP(dur, f, f_out, h)
		print(f"--- Avg Spike-Rate: {np.mean(Network.Gen.spike_count)/(dur*Network.dt)*1000} Hz")
		print(Network.Gen.J)

		if trial%plot_int == 0 or trial == trials-1:
			x = Network.Gen.simulate(dur, f, f_out=f_out, h=h)
			# x = np.zeros(snapshot_len)
			# for t in range(0, snapshot_len):
			# 	x[t] = Network.Gen.step(f, t, f_out=f_out, h=h)

			# Network.reset_spike_time_Gen()

			y = np.zeros(snapshot_len)
			for t in range(0, snapshot_len):
				y[t] = f_out[t]

			plt.plot(x.flatten(), label="activity")
			plt.plot(y, label="function")
			plt.legend(loc="upper left")
			plt.show()


def test(Network, f, f_out, h, init_trials, snapshot_len, dur):
	Network.Gen.reset_activity()
	Network.Per.reset_activity()

	for i in range(init_trials):
		Network.Gen.simulate(dur, f, f_out=f_out, h=h)
		Network.Per.simulate(dur, f, f_out=f_out, h=h)
		# for t in range(dur):
		# 	Network.Gen.step(f, t, f_out=f_out, h=h)
		# 	Network.Per.step(f, t)

	# x = np.zeros(snapshot_len)
	y = np.zeros(snapshot_len)
	
	total_error = 0

	x = Network.Per.simulate(dur, f).flatten()

	for t in range(0, snapshot_len):
		# x[t] = Network.step(f, t)
		y[t] = f_out[t]
		total_error += (x[t]-y[t])**2 

	print(f"-- MSE for test run : {total_error/snapshot_len} --")
	print(f"-- Avg spike-Rate: {np.mean(Network.Per.spike_count)/(dur*Network.dt)*1000} Hz")

	plt.plot(x.flatten(), label="output")
	plt.plot(y, label="target")
	plt.legend(loc="upper left")
	plt.show()
