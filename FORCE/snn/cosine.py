# main_cosine.py - main script for learning cosine function
#
#	created: 8/25/2021
#	last change: 8/25/2021


from snn import *
from func import *

import numpy as np
import matplotlib.pyplot as plt

# general network parameters
N_neurons = 2000
dt = 0.00005
alpha = 10/dt#1/(dt*0.1)#5*dt

N_inputs = 1
input_dims = 1

N_outputs = 1
output_dims = 1

N_hints = 0
hint_dims = 0

# parameters
G = 0.04
Q = 0.004#10
tm = 0.01
td = 0.02
tr = 0.002
ts = 0.1
E_L = -60
v_act = -45
bias = v_act
var_J = G/np.sqrt(N_neurons)
mu_w = 0
var_w = 0
var_u = Q

# training parameters
dur = 1
dur = int(dur/dt)
trials = 4

#plotting parameters
init_trials = 1
snapshot_len = dur
plot_int = 1


N = RNN(N_neurons=N_neurons, \
		N_inputs=N_inputs, \
		input_dims=input_dims, \
		N_outputs=N_outputs, \
		output_dims=output_dims, \
		alpha=alpha, \
		dt=dt, \
		tm=tm, \
		td=td, \
		tr=tr, \
		ts=ts, \
		E_L=E_L, \
		v_act=v_act, \
		bias=bias, \
		var_J=var_J, \
		mu_w=mu_w, \
		var_w=var_w, \
		var_u=var_u)


f_out = np.zeros((dur, N_outputs, output_dims))

def Cosine(t):
	return np.sin(2*np.pi*t*5)

for t in range(f_out.shape[0]):
	f_out[t][0][0] = Cosine(t*dt)

# N.reset_activity()

# x = np.zeros(snapshot_len)
# for t in range(0, snapshot_len):
# 	x[t] = N.step()

# y = np.zeros(snapshot_len)
# for t in range(0, snapshot_len):
# 	y[t] = f_out[t]

# plt.plot(x, label="output")
# plt.plot(y, label="target")
# plt.legend(loc="upper left")
# plt.show()

training(Network=N, f_out=f_out, trials=trials, snapshot_len=snapshot_len, \
		plot_int=plot_int, dur=dur, p=50)

test(Network=N, f_out=f_out, snapshot_len=snapshot_len, dur=dur)