# main_accordian.py - main script for learning accordian function
#
#	created: 7/21/2021
#	last change: 7/21/2021


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
gg = 1
tau_g = 10
mu_Jg = 0
# var_Jg = (gg**2)/N_neurons
var_Jg = gg/np.sqrt(N_neurons)
mu_wg = 0
# var_wg = (1/N_neurons)**0.5
var_wg = 1

# task-performing parameters
gp = 1
tau_p = 10
mu_Jp = 0
# var_Jp = (gp**2)/N_neurons
var_Jp = gp/np.sqrt(N_neurons)
mu_wp = 0
# var_wp = (1/N_neurons)**0.5
var_wp = 1

# training parameters
dur = 15000
trials_i = 10
trials_w = 10

#plotting parameters
snapshot_len = 300
plot_int_i = 1
plot_int_w = 5


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

# parameters for accordian function
target_T = 300
upper = 6
lower = 2

f = [input_spike]
f_out = [Accordian(target_T, upper, lower)]
h = None 

# training(Network=N, f=f, f_out=f_out, h=h, trials_i=trials_i, trials_w=trials_w, \
# 			snapshot_len=snapshot_len, plot_int_i=plot_int_i, plot_int_w=plot_int_w, \
# 			dur=dur, dt=dt)

trials = 100
plot_int = 5

train2(Network=N, f=f, f_out=f_out, h=h, trials=trials, snapshot_len=snapshot_len, \
		plot_int=plot_int, dur=dur, p=2, dt=1)