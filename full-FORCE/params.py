# params.py - define hyperparameters for full-FORCE network
#
#	created: 7/7/2021
#	last change: 7/16/2021

N_neurons = 500
alpha = 1
dt = 1

# target-generating parameters
gg = 1.5
tau_g = 10
mu_Jg = 0
var_Jg = (gg**2)/N_neurons
mu_wg = 0
var_wg = (1/N_neurons)**0.5

# task-performing parameters
gp = 1.5
tau_p = 10
mu_Jp = 0
var_Jp = (gp**2)/N_neurons
mu_wp = 0
var_wp = (1/N_neurons)**0.5