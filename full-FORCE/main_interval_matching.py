# main_interval_matching.py - main script for training interval matching task
#
#	created: 7/21/2021
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
trials = 1500
# spike_interval_upper = 2100
# spike_interval_lower = 100
spike_interval_upper = 1000
spike_interval_lower = 700
# mu_pause = 2400
mu_pause = 1400
save_int = 5

#plotting parameters
init_trials = 5
plot_int = 1500


def trainining_interval(Network, init_trials, trials, upper, lower, mu_pause, plot_int, save_int, dt):

	# initialize
	print("----- TRAINING -----")
	print("--- Initializing ---")

	np.save("model\\u_in\\u_in.npy", Network.Per.u_in)

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
			Network.Gen.step(f, t, f_out=f_out, h=h, dt=dt)
			Network.Per.step(f, t, dt=dt)

	print("--- Starting training ---")
	# train
	for trial in range(trials):
		print(f"- Trial {trial+1} -")
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

		# train
		Network.train_once(dur, f, f_out, h, 2, dt)

		if trial%plot_int == 0 or trial == trials-1:
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

			# run trial
			x = np.zeros((dur))
			for t in range(dur):
				x[t] = Network.Per.step(f, t, dt=dt)

			plt.plot(x[:dur-p], label="output")
			plt.plot(f_out[:(dur-p), 0, 0], label="target")
			plt.plot(f[:(dur-p), 0, 0], label="input")
			plt.plot(h[:(dur-p), 0, 0], label="hint")
			plt.legend(loc="upper left")
			plt.show()

		if trial%save_int == 0 or trial == trials-1:
			np.save(f"model\\w\\w_{trial}.npy", Network.Per.w)
			np.save(f"model\\J\\J_{trial}.npy", Network.Per.J)
			np.save(f"model\\P\\P_{trial}.npy", Network.P)


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

trainining_interval(Network=N, \
					init_trials=init_trials, \
					trials=trials, \
					upper=spike_interval_upper, \
					lower=spike_interval_lower, \
					mu_pause=mu_pause, \
					plot_int=plot_int, \
					save_int=save_int, \
					dt=dt)