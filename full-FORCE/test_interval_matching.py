# test_interval_matching.py - main script for testing interval matching task
#
#	created: 7/28/2021
#	last change: 7/29/2021

from network import *
from func import *

import numpy as np
import matplotlib.pyplot as plt

import random

# general network parameters
N_neurons = 1000
alpha = 1
dt = 1

N_inputs = 1
input_dims = 1

N_outputs = 1
output_dims = 1

N_hints = 1
hint_dims = 1

# target-generating parameters
gg = 1.5
tau_g = 10
mu_Jg = 0
var_Jg = gg/np.sqrt(N_neurons)
mu_wg = 0
var_wg = 1

# task-performing parameters
gp = 1.5
tau_p = 10
mu_Jp = 0
var_Jp = gp/np.sqrt(N_neurons)
mu_wp = 0
var_wp = 1

# training parameters
spike_interval_upper = 1000
spike_interval_lower = 700

mu_pause = 1400
save_int = 5

#plotting parameters
init_trials = 5
plot_int = 10000

def test(N, init_trials, upper, lower, trial_num, spiketimes, mu_pause):
	N.Per.u_in = np.load(f"trained_model\\u_in\\u_in.npy")
	N.Per.J = np.load(f"trained_model\\J\\J_{trial_num}.npy")
	N.Per.w = np.load(f"trained_model\\w\\w_{trial_num}.npy")
	N.P = np.load(f"trained_model\\P\\P_{trial_num}.npy")

	print("-- Initializing --")

	for trial in range(init_trials):
		# choose spike randomly
		r = int(random.uniform(lower, upper))
		# create functions for trial
		targ_func = IntMatchFunc(2*r)
		inp_func = IntMatchInput(r, 50)
		hint_func = Hint(r)

		# get pause duration
		p = int(np.random.exponential(mu_pause))

		dur = 2*r + p + 300 # add buffer of 200

		# create function arrays
		f_out = np.zeros((dur, 1, 1))
		f = np.zeros((dur, 1, 1))
		h = np.zeros((dur, 1, 1))

		for t in range(dur):
			f_out[t][0][0] = targ_func(t)
			f[t][0][0] = inp_func(t)
			h[t][0][0] = hint_func(t)

		# run network
		for t in range(dur):
			N.Gen.step(f, t, f_out=f_out, h=h, dt=dt)
			N.Per.step(f, t, dt=dt)

	print("-- Plotting:")
	for spike in spiketimes:

		print(f"\tSpiketime = {spike}")
		
		targ_func = IntMatchFunc(2*spike)
		inp_func = IntMatchInput(spike, 50)
		hint_func = Hint(spike)

		# get pause duration
		p = int(np.random.exponential(mu_pause))

		dur = 2*spike + p + 300 # add buffer of 200

		# create function arrays
		f_out = np.zeros((dur, 1, 1))
		f = np.zeros((dur, 1, 1))
		h = np.zeros((dur, 1, 1))

		for t in range(dur):
			f_out[t][0][0] = targ_func(t)
			f[t][0][0] = inp_func(t)
			h[t][0][0] = hint_func(t)

		# run trial
		x = np.zeros((dur))
		for t in range(dur):
			x[t] = N.Per.step(f, t, dt=dt)

		plt.figure(spike)
		plt.plot(x[:dur-p], label="output")
		plt.plot(f_out[:(dur-p), 0, 0], label="target")
		plt.plot(f[:(dur-p), 0, 0], label="input")
		plt.plot(h[:(dur-p), 0, 0], label="hint")
		plt.legend(loc="upper left")

	plt.show()


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
		var_Jg=var_Jg, \
		var_Jp=var_Jp, \
		mu_wg=mu_wg, \
		mu_wp=mu_wp, \
		var_wg=var_wg, \
		var_wp=var_wp, \
		hints=True, \
		N_hints=N_hints, \
		hint_dims=hint_dims)

spiketimes = [700, 750, 800, 850, 900, 950, 1000]
trial_num = 300#750

test(N, init_trials, spike_interval_upper, spike_interval_lower, trial_num, spiketimes, mu_pause)