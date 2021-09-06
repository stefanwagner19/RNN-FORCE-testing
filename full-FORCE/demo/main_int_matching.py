from snn import *
from func import *

import numpy as np
import matplotlib.pyplot as plt

import random

# general network parameters
N_neurons = 1000
dt = 0.00005
alpha = 100*dt

N_inputs = 1
input_dims = 1

N_outputs = 1
output_dims = 1

N_hints = 0
hint_dims = 0

# parameters
gg = gp = 1.5
Qg = Qp = 0.01
tm_g = tm_p = 0.01
td_g = td_p = 0.02
tr_g = tr_p = 0.002
ts_g = ts_p = 0.01
E_Lg = E_Lp = -60
v_actg = v_actp = -45
bias_g = bias_p = v_actg
var_Jg = var_Jp = gg/np.sqrt(N_neurons)
mu_wg = mu_wp = 0
var_wg = var_wp = 1

# training parameters
trials = 750
init_trials = 5
steps = 40
spike_interval_upper = 1
spike_interval_lower = 0.7
mu_pause = 1.4
a = b = 4
save_int = 5

def training_interval(Network, init_trials, trials, upper, lower, mu_pause, a, b, save_int, dt, p):

	# initialize
	print("----- TRAINING -----")
	print("--- Initializing ---")

	np.save("models\\snn\\int_match\\u_in\\u_in.npy", Network.Per.u_in)

	for trial in range(init_trials):
		# choose random spike time
		r = np.random.uniform(lower, upper)

		# choose random pause
		p = np.random.exponential(mu_pause)

		f_out = create_beta(a, b, 2*r, 2*r + p + 0.25, dt)

		inp_func = IntMatchInput(r, 0.05, dt)
		hint_func = Hint(r)

		f_in = np.zeros(f_out.shape)
		h = np.zeros(f_out.shape)

		for t in range(len(f_out)):
			f_in[t][0][0] = inp_func(t*dt)
			h[t][0][0] = hint_func(t*dt)

		for t in range(len(f_out)):
			Network.Gen.step(f_in, t, f_out, h)
			Network.Per.step(f_in, t)


	print("STARTING TRAINING")
	# start training
	for trial in range(trials):
		print(f"- Trial {trial+1} -")

		# choose random spike time
		r = np.random.uniform(lower, upper)

		# choose random pause
		p = np.random.exponential(mu_pause)

		f_out = create_beta(a, b, 2*r, 2*r + p + 0.25, dt)

		inp_func = IntMatchInput(r, 0.05, dt)
		hint_func = Hint(r)

		f_in = np.zeros(f_out.shape)
		h = np.zeros(f_out.shape)

		for t in range(len(f_out)):
			f_in[t][0][0] = inp_func(t*dt)
			h[t][0][0] = hint_func(t*dt)

		Network.train_once(len(f_out), f_in, f_out, h, p)

		if trial%save_int == 0 or trial == trials-1:
			np.save(f"models\\snn\\int_match\\w\\w_{trial}.npy", Network.Per.w)
			np.save(f"models\\snn\\int_match\\J\\J_{trial}.npy", Network.Per.J)
			np.save(f"models\\snn\\int_match\\P\\P_{trial}.npy", Network.P)


N = S_RNN(N_neurons=N_neurons, \
		N_inputs=N_inputs, \
		input_dims=input_dims, \
		N_outputs=N_outputs, \
		output_dims=output_dims, \
		alpha=alpha, \
		gg=gg, \
		gp=gp, \
		Qg=Qg, \
		Qp=Qp, \
		dt=dt, \
		tm_g=tm_g, \
		tm_p=tm_p, \
		td_g=td_g, \
		td_p=td_p, \
		tr_g=tr_g, \
		tr_p=tr_p, \
		ts_g=ts_g, \
		ts_p=ts_p, \
		E_Lg=E_Lg, \
		E_Lp=E_Lp, \
		v_actg=v_actg, \
		v_actp=v_actp, \
		bias_g=bias_g, \
		bias_p=bias_p, \
		var_Jg=var_Jg, \
		var_Jp=var_Jp, \
		mu_wg=mu_wg, \
		mu_wp=mu_wp, \
		var_wg=var_wg, \
		var_wp=var_wp, \
		hints=False, \
		N_hints=N_hints, \
		hint_dims=hint_dims)

# # example plot
# # choose random spike time
# r = np.random.uniform(0.7, 1)

# # choose random pause
# p = np.random.exponential(mu_pause)

# f_out = create_beta(4, 4, 2*r, 2*r + p + 0.25, dt)

# inp_func = IntMatchInput(r, 0.05, dt)
# hint_func = Hint(r)

# f_in = np.zeros(f_out.shape)
# h = np.zeros(f_out.shape)

# for t in range(len(f_out)):
# 	f_in[t][0][0] = inp_func(t*dt)
# 	h[t][0][0] = hint_func(t*dt)

# z = np.zeros(f_out.shape)
# y = np.zeros(f_out.shape)
# lin = np.linspace(0, len(f_out)*dt, len(f_out)).flatten()

# for t in range(len(f_out)):
# 	z[t] = N.Gen.step(f_in, t, f_out, h)
# 	y[t] = N.Per.step(f_in, t)

# print("r:",r)
# print("p:",p)

# plt.plot(lin, z.flatten(), label="output")
# plt.plot(lin, y.flatten(), label="Gen")
# plt.plot(lin, f_out.flatten(), label="target")
# plt.plot(lin, f_in.flatten(), label="input")
# plt.plot(lin, h.flatten(), label="hint")
# plt.legend(loc="upper left")
# plt.show()

training_interval(Network=N, \
					init_trials=init_trials, \
					trials=trials, \
					upper=spike_interval_upper, \
					lower=spike_interval_lower, \
					mu_pause=mu_pause, \
					a=a, \
					b=b, \
					save_int=save_int, \
					dt=dt, \
					p=steps)