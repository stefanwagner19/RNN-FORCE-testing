# network.py - define network for full-FORCE learning
#
#	created: 7/7/2021
#	last change: 7/16/2021
#
# TODO: only works with dt=1


import numpy as np


class Subnetwork(object):
	''' subnetwork for constructing Target-Generating and Task-Predicting network '''
	def __init__(self, N_neurons, N_inputs, input_dims, N_outputs, output_dims, g, tau, mu_J, var_J, \
					mu_w, var_w, f_out=False, N_hints=None, hint_dims=None):

		self.N_neurons = N_neurons

		self.N_inputs = N_inputs
		self.input_dims = input_dims

		self.N_outputs = N_outputs
		self.output_dims = output_dims

		self.g = g
		self.tau = tau
		self.mu_J = mu_J
		self.var_J = var_J
		self.mu_w = mu_w
		self.var_w = var_w

		self.f_out = f_out

		if self.f_out:
			self.N_hints = N_hints
			self.hint_dims = hint_dims

		# set activities and rates
		# self.x_0 = np.random.randn(self.N_neurons)*self.var_J
		self.x_0 = np.zeros((self.N_neurons))
		self.r_0 = np.tanh(self.x_0)
		self.x = self.x_0.copy()
		self.r = self.r_0.copy()

		# initialize internal connection matrix J
		self.J = np.random.normal(self.mu_J, self.var_J, (self.N_neurons, self.N_neurons))
		# np.fill_diagonal(self.J, 0.0)

		# initialize output weights w
		self.w = np.random.normal(self.mu_w, self.var_w, (self.N_outputs, self.output_dims, self.N_neurons))

		# initialize input weights u
		self.u_in = np.random.uniform(-1, 1, (self.N_inputs, self.input_dims, self.N_neurons))*50

		if self.f_out:
			self.u_out = np.random.uniform(-1, 1, (self.N_outputs, self.output_dims, self.N_neurons))*50
			if self.N_hints != None:
				self.u_h = np.random.uniform(-1, 1, (self.N_hints, self.hint_dims, self.N_neurons))*50


	def reset_activity(self):
		self.x = self.x_0
		self.r = self.r_0


	def get_error_term(self, t, f_out=None, h=None):
		''' calculate part of the error term for training '''
		sum_out = np.zeros((self.N_neurons))
		# calculate activations caused by feedback
		if f_out != None:
			for i in range(len(f_out)):
				sum_out += np.dot(self.u_out, f_out[i](t)).flatten()

		sum_h = np.zeros((self.N_neurons))
		# calculate activations caused by hints
		if h != None:
			for i in range(len(h)):
				sum_h += np.dot(self.u_h, h[i](t)).flatten()

		return np.dot(self.J, self.r) + sum_out + sum_h


	def step(self, f, t, f_out=None, h=None, dt=1):
		sum_f = np.zeros((self.N_neurons))
		# calculate activations caused by inputs
		for i in range(len(f)):
			sum_f += np.dot(self.u_in[i], f[i](t)).flatten()

		dxdt = (-self.x + sum_f + self.get_error_term(t, f_out, h)) / self.tau

		self.x += (dxdt*dt)
		self.r = np.tanh(self.x)

		y = np.zeros((self.N_outputs, self.output_dims))
		for i in range(self.N_outputs):
			y[i] = np.dot(self.w[i], self.r)


		# return np.dot(self.r, self.w)
		return y



class RNN(object):
	''' RNN class for full-FORCE learning '''
	def __init__(self, N_neurons, N_inputs, input_dims, N_outputs, output_dims, alpha, gg, gp, tau_g, tau_p, \
					mu_Jg, mu_Jp, var_Jg, var_Jp, mu_wg, mu_wp, var_wg, var_wp, N_hints=None, hint_dims=None):

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
		self.mu_Jg = mu_Jg
		self.var_Jg = var_Jg
		self.mu_wg = mu_wg
		self.var_wg = var_wg

		# task-performing network parameters
		self.gp = gp
		self.tau_p = tau_p
		self.mu_Jp = mu_Jp
		self.var_Jp = var_Jp
		self.mu_wg = mu_wg
		self.var_wg = var_wg

		# initialize matrices P for training
		self.P_i = np.identity(N_neurons)/alpha
		self.P_w = np.identity(N_neurons)/alpha

		''' create subnetworks '''

		# target-generating network
		self.Gen = Subnetwork(N_neurons=N_neurons, N_inputs=N_inputs, input_dims=input_dims, N_outputs=N_outputs, \
								output_dims=output_dims, g=gg, tau=tau_g, mu_J=mu_Jg, var_J=var_Jg, mu_w=mu_wg, \
								var_w=var_wg, f_out=True, N_hints=N_hints, hint_dims=hint_dims)

		# task-performing network
		self.Per = Subnetwork(N_neurons=N_neurons, N_inputs=N_inputs, input_dims=input_dims, N_outputs=N_outputs, \
								output_dims=output_dims, g=gp, tau=tau_p, mu_J=mu_Jp, var_J=var_Jp, mu_w=mu_wp, \
								var_w=var_wp)


	def step(self, f, t):
		return self.Per.step(f, t)


	def internal_training(self, dur, f, f_out, h=None, dt=1):
		''' train task performers internal weights J to internalize output (& hints) for one trial '''

		self.Gen.reset_activity()
		self.Per.reset_activity()

		for t in range(dur):

			e = self.Per.get_error_term(t) - self.Gen.get_error_term(t, f_out=f_out, h=h)

			''' RLS algorithm '''
			denom = 1 + np.matmul(np.transpose(self.Per.r), np.matmul(self.P_i, self.Per.r)) # check
			dPdt = -1 * np.outer(np.dot(self.P_i, self.Per.r), np.dot(np.transpose(self.Per.r), self.P_i)) / denom

			self.P_i += dPdt*dt

			dJdt = -1 * np.outer(np.transpose(e), np.dot(self.P_i, self.Per.r))
			self.Per.J += dJdt*dt

			self.Gen.step(f, t, f_out=f_out, h=h, dt=dt)
			self.Per.step(f, t, dt=dt)


	def output_training(self, dur, f, f_out, dt=1):
		''' train task performers output weights w to produce correct output for one trial '''

		self.Per.reset_activity()

		total_eminus = np.zeros((self.N_outputs))
		total_eplus = np.zeros((self.N_outputs))

		for t in range(1, dur+1):

			e_minus = np.zeros((self.N_outputs))

			y = np.zeros((self.N_outputs, self.output_dims))

			for i in range(self.N_outputs):

				y[i] = np.dot(self.Per.w[i], self.Per.r)

			for i in range(self.N_outputs):

				e_minus[i] = y[i] - f_out[i](t)


			''' RLS algorithm '''
			denom = 1 + np.dot(np.transpose(self.Per.r), np.dot(self.P_w, self.Per.r))
			dPdt = -1 * np.outer(np.dot(self.P_w, self.Per.r), np.dot(np.transpose(self.Per.r), self.P_w)) / denom
			self.P_w += dPdt*dt

			dwdt = np.zeros((self.N_outputs, self.output_dims, self.N_neurons))
			x = np.dot(self.P_w, self.Per.r)

			for i in range(self.N_outputs):
				dwdt[i] = -1 * np.dot(np.transpose(e_minus[i]), x)

			self.Per.w += dwdt*dt

			e_plus = np.zeros((self.N_outputs))

			for i in range(self.N_outputs):
				e_plus += np.dot(self.Per.w[i], self.Per.r) - f_out[i](t)

			total_eminus += abs(e_minus)
			total_eplus += abs(e_plus)

			# training feedback
			if t%100 == 0:
				print(f"--- Step {t} ---")
				print(f"Avg e_minus = {total_eminus/100}")
				print(f"Avg e_plus = {total_eplus/100}\n")

				total_eplus = np.zeros((self.N_outputs))
				total_eminus = np.zeros((self.N_outputs))

			self.Per.step(f, t, dt=dt)
