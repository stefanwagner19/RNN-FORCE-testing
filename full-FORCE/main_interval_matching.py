# main_interval_matching.py - main script for interval matching task
#
#	created: 7/21/2021
#	last change: 7/21/2021


from network import *
from func import *

import numpy as np
import matplotlib.pyplot as plt

# general network parameters
N_neurons = 200
alpha = 10
dt = 1

N_inputs = 2
input_dims = 1

N_outputs = 1
output_dims = 1

N_hints = 1
hint_dims = 1

# target-generating parameters
gg = 5.5
tau_g = 10
mu_Jg = 0
var_Jg = (gg**2)/N_neurons
mu_wg = 0
var_wg = (1/N_neurons)**0.5

# task-performing parameters
gp = 5.5
tau_p = 10
mu_Jp = 0
var_Jp = (gp**2)/N_neurons
mu_wp = 0
var_wp = (1/N_neurons)**0.5

# training parameters
dur = 1200
trials_i = 10
trials_w = 10

#plotting parameters
snapshot_len = 1200
plot_int_i = 5
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

spike = 400
f = [IntMatchFunc(0), IntMatchFunc(spike)]
f_out = [IntMatchFunc(spike*2)]
h = [Hint(spike)] 

training(Network=N, f=f, f_out=f_out, h=h, trials_i=trials_i, trials_w=trials_w, \
			snapshot_len=snapshot_len, plot_int_i=plot_int_i, plot_int_w=plot_int_w, \
			dur=dur, dt=dt)