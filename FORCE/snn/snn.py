# snn.py - define spiking network for FORCE learning
#
# 	created: 8/25/2021
#	last change: 8/25/2021


import math
import numpy as np
from scipy import sparse

class RNN(object):
	''' subnetwork for constructing Target-Generating and Task-Predicting network '''
	def __init__(self, N_neurons, N_inputs, input_dims, N_outputs, output_dims, alpha, dt, tm, td, tr, ts, E_L, v_act, bias, \
					var_J, mu_w, var_w, var_u):

		self.N_neurons = N_neurons

		self.N_inputs = N_inputs
		self.input_dims = input_dims

		self.N_outputs = N_outputs
		self.output_dims = output_dims


		self.alpha = alpha
		self.dt = dt

		self.tm = tm
		self.td = td
		self.tr = tr
		self.ts = ts
		self.E_L = E_L
		self.v_act = v_act
		self.bias = bias

		self.var_J = var_J
		
		self.mu_w = mu_w
		self.var_w = var_w

		self.var_u = var_u

		# set membrane potential and spikes
		self.v_mem = np.zeros((self.N_neurons))
		self.spikes = np.zeros((self.N_neurons))

		# set arrays for filtered firing rates
		self.r = np.zeros((self.N_neurons))
		self.h = np.zeros((self.N_neurons))

		# countdown for refractory period
		self.spike_timer = np.zeros((self.N_neurons))

		# keep track of number of spikes fired
		self.spike_count = np.zeros((self.N_neurons), dtype=np.int32)

		# initiate storage variables for current
		self.s = np.zeros((self.N_neurons))
		self.hr = np.zeros((self.N_neurons))

		# initialize internal connection matrix J
		self.J = np.array(sparse.random(self.N_neurons, self.N_neurons, 1, data_rvs=np.random.randn).todense()) * var_J
		# self.J = np.array(sparse.random(self.N_neurons, self.N_neurons, 1, data_rvs=np.random.randn).todense())

		# initialize output weights w
		self.w = np.random.normal(self.mu_w, self.var_w, (self.N_outputs, self.output_dims, self.N_neurons))

		# initialize input weights u
		self.u = np.random.uniform(-1, 1, (self.input_dims, self.N_neurons, self.N_inputs)) * var_u

		self.P = np.identity(N_neurons)/alpha


	def reset_activity(self):
		self.v_mem = np.random.randn((self.N_neurons))*(30 - self.E_L)
		# self.v_mem = np.random.uniform(self.E_L, self.v_act, (self.N_neurons))
		
		# self.spikes = np.random.randint(2, size=(self.N_neurons))

		self.s = np.zeros((self.N_neurons))
		self.h = np.zeros((self.N_neurons))
		self.r = np.zeros((self.N_neurons))
		self.hr = np.zeros((self.N_neurons))


	def step(self):

		# update current
		I = self.s + self.bias# + np.matmul(self.u, np.matmul(self.w, self.r)).flatten()

		# update voltage and spikes
		dv = (self.spike_timer <= 0)*(-self.v_mem + I)/self.tm
		self.v_mem = self.v_mem + self.dt*dv

		self.spikes = (self.v_mem >= self.v_act).astype(int)
		self.spike_timer[self.spikes != 0] = self.tr

		# update spike timer for refractory period
		self.spike_timer -= self.dt

		# double exponential filter for synaptic current
		self.s = self.s*math.exp(-self.dt/self.ts) + (np.matmul(self.J, self.spikes) + np.matmul(self.u, np.matmul(self.w, self.r)).flatten()) / self.ts
		# self.s = self.s*math.exp(-self.dt/self.tr) + self.h*self.dt
		# self.h = self.h*math.exp(-self.dt/self.td) + (np.matmul(self.J, self.spikes)) / (self.tr*self.td)

		# double exponential filter for firing rate
		self.r = self.r*math.exp(-self.dt/self.tr) + self.hr*self.dt
		self.hr = self.hr*math.exp(-self.dt/self.td) + self.spikes/(self.tr*self.td)

		
		# print(self.v_mem + (30 - self.v_mem)*self.spikes[::5])
		# print()

		# self.v_mem = self.v_mem + (30 - self.v_mem)*self.spikes


		# self.v_mem[self.v_mem < self.E_L] = self.E_L

		self.v_mem = self.v_mem + (self.E_L - self.v_mem)*(self.v_mem >= self.v_act)

		self.spike_count += self.spikes


		return np.sum(np.matmul(self.w, self.r), 0)


	def reset_spike_count(self):
		self.spike_count = np.zeros(self.N_neurons, dtype=np.int32)


	def train_once(self, dur, f_out, p):

		self.reset_spike_count()

		for t in range(dur):

			self.step()

			if np.random.rand() < (1/p):
			
				''' RLS algorithm '''

				w_err = np.matmul(self.w, self.r) - f_out[t]

				# update output weights
				dwdt = -1 * np.sum(np.matmul(np.transpose(w_err), np.expand_dims(np.matmul(self.P, self.r), 0)), 0)
				self.w += dwdt#*self.dt

				# update correlation matrix
				denom = 1 + np.matmul(np.transpose(self.r), np.matmul(self.P, self.r))
				dPdt = -1 * np.outer(np.dot(self.P, self.r), np.dot(np.transpose(self.r), self.P)) / denom
				self.P += dPdt#*self.dt

				# cd = np.matmul(self.P, np.expand_dims(self.r, 1))
				# # print(np.expand_dims(cd, 0).shape, np.transpose(w_err.shape))
				# print( np.expand_dims(self.r, 1).shape)
				# self.w = self.w - np.matmul(np.expand_dims(cd, 0), np.transpose(w_err))

				# denom = 1 + (np.matmul(np.transpose(self.r), cd))
				# self.P = self.P - np.matmul(cd, np.transpose(cd)) / denom