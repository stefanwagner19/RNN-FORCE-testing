# main.py - main script for full-FORCE training
#
#	created: 7/8/2021
#	last change: 7/16/2021

from network import *
from params import *

import numpy as np
import math
import random

import matplotlib.pyplot as plt

class IntMatchFunc(object):
	def __init__(self, spiketime):
		self.spike = spiketime
		self.std = 50

	def __call__(self, x):
		if abs(x - self.spike) < 150:
			return 1/(self.std * math.sqrt(2 * math.pi)) * math.exp( - (x - self.spike)**2 / (2 * self.std**2)) * 30
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


def accordian_out(t):
	dw = 4*math.pi/1000
	if t <= 1000:
		w = 2*math.pi+t*dw
		return math.sin(w*t/1000)
	if t > 1000:
		w = 6*math.pi-(t-1000)*dw
		return -math.sin(w*(2000-t)/1000)

def accordian_in(t):
	if t <= 1:
		return 0.3
	else:
		return 0


def cosine(t):
	return 1.5*np.cos(2*np.pi*10*t/1000)


def training(Network, trials_i, trials_w, dur=1200, dt=1, lower=100, upper=500):

	print("<--- STARTING TRAINING --->\n")

	print("internal training ...\n")

	for trial in range(trials_i):
		print(f"internal trial {trial+1}")
		# create functions and hint
		spike = random.randint(lower, upper)
		f = [IntMatchFunc(0), IntMatchFunc(spike)]
		h = [Hint(spike)]
		f_out = [IntMatchFunc(spike*2)]
		
		# training for one trial
		Network.internal_training(dur, f, f_out, h, dt)


	print("output training ...\n")

	for trial in range(trials_w):
		print(f"output trial {trial+1}")
		# create functions
		spike = random.randint(lower, upper)
		f = [IntMatchFunc(0), IntMatchFunc(spike)]
		f_out = [IntMatchFunc(spike*2)]

		# training for one trial
		Network.output_training(dur, f, f_out, dt)


ACCORDIAN = True
BASIC = False

if not ACCORDIAN and not BASIC:

	N_inputs = 2
	input_dims = 1

	N_outputs = 1
	output_dims = 1

	N_hints = 1
	hint_dims = 1

	dur = 1200
	trials_i = 10
	trials_w = 10


	N = RNN(N_neurons=N_neurons, \
			N_inputs=N_inputs, \
			input_dims=input_dims, \
			N_outputs=N_outputs, \
			output_dims=output_dims, \
			alpha=alpha, \
			gg=gg, \
			gp=gp, \
			tau_g=tau_g, \
			tau_p=tau_p, \
			mu_Jg=mu_Jg, \
			mu_Jp=mu_Jp, \
			var_Jg=var_Jg, \
			var_Jp=var_Jp, \
			mu_wg=mu_wg, \
			mu_wp=mu_wp, \
			var_wg=var_wg, \
			var_wp=var_wp, \
			N_hints=N_hints, \
			hint_dims=hint_dims)

	training(N, trials_i, trials_w, dur, dt)

	spike = 400
	f = [IntMatchFunc(0), IntMatchFunc(spike)]
	f_out = [IntMatchFunc(spike*2)]
	h = [Hint(spike)] 

	x = np.zeros((dur))
	y = np.zeros((dur))

	N.Per.w = N.Gen.w

	for t in range(dur):
		x[t] = N.step(f, t)
		# y[t] = N.Gen.step(f, t, f_out, h)

	plt.plot(x)
	# plt.show()
	# plt.plot(y)
	plt.show()

	# x = np.zeros((2, dur))

	# for t in range(dur):
	# 	x[0, t] = f[0](t)
	# 	x[1, t] = f[1](t)

	# plt.plot(x[0])
	# plt.show()

	# plt.plot(x[1])
	# plt.show()

elif not BASIC:

	N_inputs = 1
	input_dims = 1

	N_outputs = 1
	output_dims = 1


	dur = 2000
	trials_i = 50
	trials_w = 50


	N = RNN(N_neurons=N_neurons, \
			N_inputs=N_inputs, \
			input_dims=input_dims, \
			N_outputs=N_outputs, \
			output_dims=output_dims, \
			alpha=alpha, \
			gg=gg, \
			gp=gp, \
			tau_g=tau_g, \
			tau_p=tau_p, \
			mu_Jg=mu_Jg, \
			mu_Jp=mu_Jp, \
			var_Jg=var_Jg, \
			var_Jp=var_Jp, \
			mu_wg=mu_wg, \
			mu_wp=mu_wp, \
			var_wg=var_wg, \
			var_wp=var_wp)

	f = [accordian_in]
	f_out = [accordian_out]

	x = np.zeros((dur))

	for t in range(dur):
		x[t] = f_out[0](t)

	plt.plot(x)
	plt.show()

	print("<--- STARTING TRAINING --->\n")

	print("internal training ...\n")

	for trial in range(trials_i):
		print(f"internal trial {trial+1}")
		# training for one trial
		N.internal_training(dur, f, f_out)


	print("output training ...\n")

	for trial in range(trials_w):
		print(f"output trial {trial+1}")
		# training for one trial
		N.output_training(dur, f, f_out)

	x = np.zeros((dur))

	for t in range(dur):
		x[t] = N.step(f, t)
		# y[t] = N.Gen.step(f, t, f_out, h)

	plt.plot(x)
	# plt.show()
	# plt.plot(y)
	plt.show()

else:

	N_inputs = 1
	input_dims = 1

	N_outputs = 1
	output_dims = 1


	dur = 2000
	trials_i = 10
	trials_w = 10


	N = RNN(N_neurons=N_neurons, \
			N_inputs=N_inputs, \
			input_dims=input_dims, \
			N_outputs=N_outputs, \
			output_dims=output_dims, \
			alpha=alpha, \
			gg=gg, \
			gp=gp, \
			tau_g=tau_g, \
			tau_p=tau_p, \
			mu_Jg=mu_Jg, \
			mu_Jp=mu_Jp, \
			var_Jg=var_Jg, \
			var_Jp=var_Jp, \
			mu_wg=mu_wg, \
			mu_wp=mu_wp, \
			var_wg=var_wg, \
			var_wp=var_wp)

	f = [accordian_in]
	f_out = [cosine]

	x = np.zeros((dur))

	for t in range(dur):
		x[t] = f_out[0](t)

	plt.plot(x)
	plt.show()

	# for t in range(dur):
	# 	x[t] = N.step(f_out, t)
	# 	# y[t] = N.Gen.step(f, t, f_out, h)

	# plt.plot(x)
	# plt.show()

	print("<--- STARTING TRAINING --->\n")

	print("internal training ...\n")

	for trial in range(trials_i):
		print(f"internal trial {trial+1}")
		# training for one trial
		N.internal_training(dur, f, f_out)


	print("output training ...\n")

	for trial in range(trials_w):
		print(f"output trial {trial+1}")
		# training for one trial
		N.output_training(dur, f, f_out)

	x = np.zeros((dur))
	y = np.zeros((dur))

	for t in range(dur):
		x[t] = N.step(f, t)
		# y[t] = N.Gen.step(f, t, f_out)

	plt.plot(x)
	# plt.show()
	# plt.plot(y)
	plt.show()