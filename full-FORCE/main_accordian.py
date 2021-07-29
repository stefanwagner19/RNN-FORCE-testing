# main_accordian.py - main script for learning accordian function
#
#	created: 7/21/2021
#	last change: 7/29/2021


from network import *
from func import *

import numpy as np
import matplotlib.pyplot as plt

# general network parameters
N_neurons = 300
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
var_Jp = gp/np.sqrt(N_neurons)
mu_wp = 0
var_wp = 1

# training parameters
dur =2100
trials = 100

#plotting parameters
init_trials = 20
snapshot_len = 2100
plot_int = 100


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
		hints=False, \
		N_hints=N_hints, \
		hint_dims=hint_dims)

# parameters for accordian function
target_T = 2100
upper = 6
lower = 2

f = np.zeros((dur, N_inputs, input_dims))

for t in range(dur):
	f[t][0][0] = input_spike(t)

accord = Accordian(target_T, upper, lower)
f_out = np.zeros((dur, N_outputs, output_dims))

for t in range(f_out.shape[0]):
	f_out[t][0][0] = accord(t)

h = None 

training(Network=N, f=f, f_out=f_out, h=h, trials=trials, snapshot_len=snapshot_len, \
		plot_int=plot_int, dur=dur, p=2, dt=1)

test(Network=N, f=f, f_out=f_out, h=h, init_trials=init_trials, snapshot_len=snapshot_len, dur=dur, dt=dt)