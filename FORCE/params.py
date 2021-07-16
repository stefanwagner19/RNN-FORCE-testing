# params.py - defines network parameters
#
#	created: 7/6/2021
#	last change: 7/16/2021

import math

# # cosine/sine
# N_neurons = 300
# N_outdims = 1
# gg = 120
# gz = 1
# p_gg = 0.1
# alpha = 1
# tau = 10

# sawtooth
N_neurons = 500
N_outdims = 1
gg = 500
gz = 1
p_gg = 0.1
alpha = 1
tau = 10

sigma_gg = 1/(p_gg*N_neurons)
sigma_w = math.sqrt(1/N_neurons)