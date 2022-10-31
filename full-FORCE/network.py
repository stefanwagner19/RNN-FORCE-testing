# network.py - define network for full-FORCE learning
#
#	created: 7/7/2021
#	last change: 7/27/2021


import numpy as np
from scipy import sparse

import matplotlib.pyplot as plt


class Subnetwork(object):
	''' subnetwork for constructing Target-Generating and Task-Predicting network '''
	def __init__(self, N_neurons, N_inputs, input_dims, N_outputs, output_dims, tau, var_J, \
					mu_w, var_w, f_out=False, hints=False, N_hints=None, hint_dims=None):

		self.N_neurons = N_neurons

		self.N_inputs = N_inputs
		self.input_dims = input_dims

		self.N_outputs = N_outputs
		self.output_dims = output_dims

		self.tau = tau
		self.var_J = var_J
		self.mu_w = mu_w
		self.var_w = var_w

		self.f_out = f_out
		self.hints = hints

		if self.f_out:
			self.N_hints = N_hints
			self.hint_dims = hint_dims

		# set activities and rates
		self.x_0 = np.zeros((self.N_neurons))
		self.r_0 = np.tanh(self.x_0)
		self.x = self.x_0.copy()
		self.r = self.r_0.copy()

		# initialize internal connection matrix J
		self.J = np.array(sparse.random(self.N_neurons, self.N_neurons, 1, data_rvs=np.random.randn).todense()) \
					* var_J

		# initialize output weights w
		self.w = np.random.normal(self.mu_w, self.var_w, (self.N_outputs, self.output_dims, self.N_neurons))

		# initialize input weights u
		self.u_in = np.random.uniform(-1, 1, (self.input_dims, self.N_neurons, self.N_inputs))

		if self.f_out:
			self.u_out = np.random.uniform(-1, 1, (self.output_dims, self.N_neurons, self.N_outputs))
			if self.N_hints != None:
				self.u_h = np.random.uniform(-1, 1, (self.hint_dims, self.N_neurons, self.N_hints))


	def reset_activity(self):
		self.x = np.random.randn((self.N_neurons))
		self.r = np.tanh(self.x)


	def get_error_term(self, t, f_out=None, h=None):
		''' calculate part of the error term for training '''

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

		return np.dot(self.J, self.r) + sum_out + sum_h


	def step(self, f, t, f_out=None, h=None, dt=1):
		# calculate activations caused by inputs
		sum_f = np.sum(np.matmul(self.u_in, f[t]), 0).flatten()
		dxdt = (-self.x + sum_f + self.get_error_term(t, f_out, h)) / self.tau

		self.x += dxdt*dt
		self.r = np.tanh(self.x)

		return np.sum(np.matmul(self.w, self.r), 0)



class RNN(object):
	''' RNN class for full-FORCE learning '''
	def __init__(self, N_neurons, N_inputs, input_dims, N_outputs, output_dims, alpha, gg, gp, tau_g, tau_p, \
					var_Jg, var_Jp, mu_wg, mu_wp, var_wg, var_wp, hints=False, N_hints=None, hint_dims=None):

		self.N_neurons = N_neurons
		self.alpha = alpha

		self.N_inputs = N_inputs
		self.input_dims = input_dims

		self.N_outputs = N_outputs
		self.output_dims = output_dims

		if N_hints != None:
			self.N_hints = N_hints
			self.hint_dims = hint_dims

		# target-generating network parameters
		self.gg = gg
		self.tau_g = tau_g
		self.var_Jg = var_Jg
		self.mu_wg = mu_wg
		self.var_wg = var_wg

		# task-performing network parameters
		self.gp = gp
		self.tau_p = tau_p
		self.var_Jp = var_Jp
		self.mu_wg = mu_wg
		self.var_wg = var_wg

		# initialize matrices P for training
		self.P = np.identity(N_neurons)/alpha

		''' create subnetworks '''

		# target-generating network
		self.Gen = Subnetwork(N_neurons=N_neurons, N_inputs=N_inputs, input_dims=input_dims, N_outputs=N_outputs, \
								output_dims=output_dims, tau=tau_g, var_J=var_Jg, mu_w=mu_wg, var_w=var_wg, \
								f_out=True, hints=hints, N_hints=N_hints, hint_dims=hint_dims)

		# task-performing network
		self.Per = Subnetwork(N_neurons=N_neurons, N_inputs=N_inputs, input_dims=input_dims, N_outputs=N_outputs, \
								output_dims=output_dims, tau=tau_p, var_J=var_Jp, mu_w=mu_wp, var_w=var_wp)


	def step(self, f, t):
		return self.Per.step(f, t)


	def train_once(self, dur, f, f_out, h, p, dt):

		x = np.zeros(dur)
		y = np.zeros(dur)

		W = self.Per.w

		for t in range(dur):

			if np.random.rand() < (1/p):
			
				''' RLS algorithm '''
				J_err = (self.Per.get_error_term(t) - self.Gen.get_error_term(t, f_out=f_out, h=h))

				w_err = (np.sum(np.matmul(self.Per.w, self.Per.r), 0) - f_out[t])

				# update correlation matrix
				denom = 1 + np.matmul(np.transpose(self.Per.r), np.matmul(self.P, self.Per.r))
				dPdt = -1 * np.outer(np.dot(self.P, self.Per.r), np.dot(np.transpose(self.Per.r), self.P)) / denom
				self.P += dPdt*dt

				print(self.P[0][0])

				# update internal weights
				dJdt = -1 * np.matmul(np.transpose(np.expand_dims(J_err, 0)), np.expand_dims(np.matmul(self.P, self.Per.r), 0)) 
				self.Per.J += dJdt*dt

				# update output weights
				dwdt = -1 * np.sum(np.matmul(np.transpose(w_err), np.expand_dims(np.matmul(self.P, self.Per.r), 0)), 0)
				self.Per.w += dwdt*dt

			x[t] = np.matmul(W, self.Per.r).flatten()
			y[t] = np.matmul(W, self.Gen.r).flatten()

			self.Gen.step(f, t, f_out=f_out, h=h, dt=dt)
			self.Per.step(f, t, dt=dt)

		plt.plot(x, label="Per")
		plt.plot(y, label="Gen")
		plt.legend(loc="upper left")
		plt.show()