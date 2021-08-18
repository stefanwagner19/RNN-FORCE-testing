# main_cosine.py - main script for learning cosine function
#
#	created: 8/14/2021
#	last change: 8/14/2021


from snn import *
from func import *

import numpy as np
import matplotlib.pyplot as plt

# general network parameters
N_neurons = 100
dt = 1
alpha = 0.2*dt

N_inputs = 1
input_dims = 1

N_outputs = 1
output_dims = 1

N_hints = 0
hint_dims = 0

# parameters
gg = gp = 5
tm_g = tm_p = 10
td_g = td_p = 20
tr_g = tr_p = 2
ts_g = ts_p = 10
E_Lg = E_Lp = 0
v_actg = v_actp = 1
bias_g = bias_p = v_actg
# bias_g = bias_p = 0
var_Jg = var_Jp = gg/np.sqrt(N_neurons)
mu_wg = mu_wp = 0
var_wg = var_wp = 1

# STDP parameters
eta = 0.005
tp = 20
x_offset = 0.5
# J_max = np.max(J)
# J_min = np.min(J)


# task-performing parameters
# gp = 1.5
# tm_p = 10
# td_p = 20
# tr_p = 2
# ts_p = 1
# E_Lp = -65
# v_actp = -40
# bias_p = v_actp
# # bias_p = 0
# var_Jp = gp/np.sqrt(N_neurons)
# mu_wp = 0
# var_wp = 1

# training parameters
dur = 1000
dur = int(dur/dt)
trials = 20
trials_pretrain = 10

#plotting parameters
init_trials = 10
snapshot_len = dur
plot_int = 5


N = S_RNN(N_neurons=N_neurons, \
		N_inputs=N_inputs, \
		input_dims=input_dims, \
		N_outputs=N_outputs, \
		output_dims=output_dims, \
		alpha=alpha, \
		eta=eta, \
		tp=tp, \
		x_offset=x_offset, \
		gg=gg, \
		gp=gp, \
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


f = np.zeros((dur, N_inputs, input_dims))

for t in range(dur):
	f[t][0][0] = input_spike(t*dt)

f_out = np.zeros((dur, N_outputs, output_dims))

for t in range(f_out.shape[0]):
	f_out[t][0][0] = Cosine(t*dt)

h = None 

N.Per.reset_activity()
N.Gen.reset_activity()

x = N.Per.simulate(dur, f)
z = N.Gen.simulate(dur, f, f_out=f_out, h=h)
# x = np.zeros(snapshot_len)
# z = np.zeros(snapshot_len)
# for t in range(0, snapshot_len):
# 	x[t] = N.Gen.step(f, t, f_out)
# 	z[t] = N.Per.step(f, t)

# N.reset_spike_time_Gen()
# N.reset_spike_time_Per()

y = np.zeros(snapshot_len)
for t in range(0, snapshot_len):
	y[t] = f_out[t]

plt.plot(x.flatten(), label="Gen")
plt.plot(z.flatten(), label="Per")
plt.plot(y, label="target")
plt.legend(loc="upper left")
plt.show()

STDP_pre_training(Network=N, f=f, f_out=f_out, h=h, init_trials=init_trials, trials=trials_pretrain, \
					snapshot_len=snapshot_len, plot_int=plot_int, dur=dur)

training(Network=N, f=f, f_out=f_out, h=h, init_trials=init_trials, trials=trials, snapshot_len=snapshot_len, \
					plot_int=plot_int, dur=dur, p=2)

test(Network=N, f=f, f_out=f_out, h=h, init_trials=init_trials, snapshot_len=snapshot_len, dur=dur)