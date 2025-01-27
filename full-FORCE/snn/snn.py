import math
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

class Subnetwork(object):
	''' subnetwork for constructing Target-Generating and Task-Predicting network '''
	def __init__(self, N_neurons, Q, N_inputs, input_dims, N_outputs, output_dims, dt, tm, td, tr, ts, E_L, v_act, \
					bias, var_J, mu_w, var_w, f_out=False, hints=False, N_hints=None, hint_dims=None):

		self.N_neurons = N_neurons
		self.Q = Q

		self.N_inputs = N_inputs
		self.input_dims = input_dims

		self.N_outputs = N_outputs
		self.output_dims = output_dims

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

		self.f_out = f_out
		self.hints = hints

		if self.f_out:
			self.N_hints = N_hints
			self.hint_dims = hint_dims

		# set membrane potential and spikes
		self.v_mem = np.zeros((self.N_neurons))
		self.spikes = np.zeros((self.N_neurons))

		# track total voltage
		self.v_ = np.zeros((self.N_neurons))

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
		self.u_in = np.random.uniform(-1, 1, (self.input_dims, self.N_neurons, self.N_inputs))

		if self.f_out:
			self.u_out = np.random.uniform(-1, 1, (self.output_dims, self.N_neurons, self.N_outputs))

			if self.hints:
				self.u_h = np.random.uniform(-1, 1, (self.hint_dims, self.N_neurons, self.N_hints))


	def reset_activity(self):
		self.v_mem = np.random.uniform(self.E_L, self.v_act, (self.N_neurons))
		
		self.spikes = np.random.randint(2, size=(self.N_neurons))

		self.s = np.zeros((self.N_neurons))
		self.h = np.zeros((self.N_neurons))
		self.r = np.zeros((self.N_neurons))
		self.hr = np.zeros((self.N_neurons))


	def get_incoming_currents(self, t, f_out=None, h=None):
		''' calculate part of the activities '''

		# calculate activations caused by feedback
		if self.f_out:
			sum_out = np.sum(np.matmul(self.u_out, f_out[t]), 0).flatten()
		else:
			sum_out = np.zeros((self.N_neurons))

		sum_h = np.zeros((self.N_neurons))
		# calculate activations caused by hints
		if self.hints:
			sum_h = np.sum(np.matmul(self.u_h, h[t]), 0).flatten()
		else:
			sum_h = np.zeros((self.N_neurons))

		return np.dot(self.J, self.r*self.dt) + (sum_out + sum_h)*self.Q


	def step(self, f, t, f_out=None, h=None):

		# calculate activations caused by inputs
		sum_f = np.sum(np.matmul(self.u_in, f[t]), 0).flatten()

		# double exponential filter for synaptic current
		self.s = self.s*math.exp(-self.dt/self.tr) + self.h*self.dt
		self.h = self.h*math.exp(-self.dt/self.td) + (self.get_incoming_currents(t, f_out, h) + sum_f) / (self.tr*self.td)

		# double exponential filter for firing rate
		self.r = self.r*math.exp(-self.dt/self.tr) + self.hr*self.dt
		self.hr = self.hr*math.exp(-self.dt/self.td) + self.spikes/(self.tr*self.td)

		# update current
		I = self.s + self.bias

		# update spike timer for refractory period
		self.spike_timer -= self.dt

		# update voltage and spikes
		dv = (self.spike_timer <= 0)*(-self.v_mem + I)/self.tm
		x = self.v_mem + self.dt*dv

		self.v_ += x - self.v_mem

		self.spikes = (x >= self.v_act).astype(int)
		self.spike_timer[self.spikes != 0] = self.tr
		x[x < self.E_L] = self.E_L

		self.v_mem = x + (self.E_L - x)*(x >= self.v_act)

		self.spike_count += self.spikes


		return np.sum(np.matmul(self.w, self.r*self.dt), 0)


class S_RNN(object):
	''' Spiking RNN class for full-FORCE learning '''
	def __init__(self, N_neurons, N_inputs, input_dims, N_outputs, output_dims, alpha, gg, gp, Qg, Qp, dt, tm_g, tm_p, \
					td_g, td_p, tr_g, tr_p, ts_g, ts_p, E_Lg, E_Lp, v_actg, v_actp, bias_g, bias_p, var_Jg, var_Jp, \
					mu_wg, mu_wp, var_wg, var_wp, hints=False, N_hints=None, hint_dims=None):

		self.N_neurons = N_neurons
		self.alpha = alpha

		self.N_inputs = N_inputs
		self.input_dims = input_dims

		self.N_outputs = N_outputs
		self.output_dims = output_dims

		if N_hints != None:
			self.N_hints = N_hints
			self.hint_dims = hint_dims

		self.dt = dt

		# target-generating network parameters
		self.gg = gg
		self.Qg = Qg
		self.tm_g = tm_g
		self.td_g = td_g
		self.tr_g = tr_g
		self.ts_g = ts_g
		self.E_Lg = E_Lg
		self.v_actg = v_actg
		self.bias_g = bias_g
		self.var_Jg = var_Jg
		self.mu_wg = mu_wg
		self.var_wg = var_wg

		# task-performing network parameters
		self.gp = gp
		self.Qp = Qp
		self.tm_p = tm_p
		self.td_p = td_p
		self.tr_p = tr_p
		self.ts_p = ts_p
		self.E_Lp = E_Lp
		self.v_actp = v_actp
		self.bias_p = bias_p
		self.var_Jp = var_Jp
		self.mu_wg = mu_wg
		self.var_wg = var_wg

		# initialize matrices P for training
		self.P = np.identity(N_neurons)/alpha

		''' create subnetworks '''

		# target-generating network
		self.Gen = Subnetwork(N_neurons=N_neurons, \
								Q=Qg, \
								N_inputs=N_inputs, \
								input_dims=input_dims, \
								N_outputs=N_outputs, \
								output_dims=output_dims, \
								dt=dt, \
								tm=tm_g, \
								td=td_g, \
								tr=tr_g, \
								ts=ts_g, \
								E_L=E_Lg, \
								v_act=v_actg, \
								bias=bias_g, \
								var_J=var_Jg, \
								mu_w=mu_wg, \
								var_w=var_wg, \
								f_out=True, \
								hints=hints, \
								N_hints=N_hints, \
								hint_dims=hint_dims)

		# task-performing network
		self.Per = Subnetwork(N_neurons=N_neurons, \
								Q=Qp, \
								N_inputs=N_inputs, \
								input_dims=input_dims, \
								N_outputs=N_outputs, \
								output_dims=output_dims, \
								dt=dt, \
								tm=tm_p, \
								td=td_p, \
								tr=tr_p, \
								ts=ts_p, \
								E_L=E_Lp, \
								v_act=v_actp, \
								bias=bias_p, \
								var_J=var_Jp, \
								mu_w=mu_wp, \
								var_w=var_wp)


	def step(self, f, t):
		return self.Per.step(f, t)


	def reset_spike_count(self):
		self.Per.spike_count = np.zeros(self.N_neurons, dtype=np.int32)
		self.Gen.spike_count = np.zeros(self.N_neurons, dtype=np.int32) 


	def train_once(self, dur, f, f_out, h, p):

		self.reset_spike_count()

		for t in range(dur):

			if np.random.rand() < (1/p):
			
				''' RLS algorithm '''
				J_err = (self.Per.get_incoming_currents(t) - self.Gen.get_incoming_currents(t, f_out=f_out, h=h))
				w_err = np.sum(np.matmul(self.Per.w, self.Per.r*self.dt), 0) - f_out[t]

				# update internal weights
				dJdt = -1 * np.matmul(np.transpose(np.expand_dims(J_err, 0)), np.expand_dims(np.matmul(self.P, self.Per.r*self.dt), 0)) 
				self.Per.J += dJdt

				# update output weights
				dwdt = -1 * np.sum(np.matmul(np.transpose(w_err), np.expand_dims(np.matmul(self.P, self.Per.r*self.dt), 0)), 0)
				self.Per.w += dwdt

				# update correlation matrix
				denom = 1 + np.matmul(np.transpose(self.Per.r*self.dt), np.matmul(self.P, self.Per.r*self.dt))
				dPdt = -1 * np.outer(np.dot(self.P, self.Per.r*self.dt), np.dot(np.transpose(self.Per.r*self.dt), self.P)) / denom
				self.P += dPdt

			self.Gen.step(f, t, f_out=f_out, h=h)
			self.Per.step(f, t)