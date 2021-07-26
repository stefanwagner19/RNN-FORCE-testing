# network.py - define network for FORCE learning
#
# 	created: 7/2/2021
# 	last change: 7/18/2021

import numpy as np


class Network():
	''' implementation of chaotic recurrent network for FORCE learning '''
	def __init__(self, N_neurons, output_dim, gg, gz, alpha, sigma_gg, sigma_w, p_gg, p_gz, p_w, tau):
		self.N_neurons = N_neurons
		self.output_dim = output_dim
		self.gg = gg
		self.gz = gz
		self.alpha = alpha
		self.sigma_gg = sigma_gg
		self.sigma_w = sigma_w
		self.p_gg = p_gg
		self.p_gz = p_gz
		self.p_w = p_w
		self.tau = tau

		# activity of individual neurons
		# self.x_0 = np.random.randn(self.N_neurons)*(self.sigma_gg**2)
		# self.x = self.x_0.copy()
		self.x = np.zeros((self.N_neurons))
		
		# firing rates
		# self.r_0 = np.tanh(self.x_0)
		# self.r = np.tanh(self.x)
		self.r = np.zeros((self.N_neurons))

		# initialize random internal weights
		self.Jgg = np.random.normal(0, self.sigma_gg, (self.N_neurons, self.N_neurons))
		self.Jgg[np.random.uniform(0, 1, (self.N_neurons, self.N_neurons)) < 1-self.p_gg] = 0
		# np.fill_diagonal(self.Jgg, 0.0)

		# initialize random feedback weights
		self.Jgz = np.random.uniform(-1, 1, (self.output_dim, self.N_neurons))
		self.Jgz[np.random.uniform(0, 1, (self.output_dim, self.N_neurons)) < 1-self.p_gz] = 0
		
		# initialize random output weights
		self.w = np.random.normal(0, self.sigma_w, self.N_neurons)
		self.w[np.random.uniform(0, 1, self.N_neurons) < 1-self.p_w] = 0

		# matrix P for RLS algorithm
		self.P = np.identity(self.N_neurons)/self.alpha


	def reset_activations(self):
		# self.x = self.x_0
		# self.r = self.r_0
		self.x = np.zeros((self.N_neurons))
		self.r = np.zeros((self.N_neurons))


	def step(self, z, dt):
		''' simulate one step of network given input f and stepsize dt '''
		# calculate netork activity and update activity and firing rates
		dxdt = (-self.x + self.gg*np.dot(self.Jgg, self.r) + self.gz*np.dot(self.Jgz, z))/self.tau
		# dxdt = (-self.x + np.dot(self.Jgg, self.r) + np.dot(self.Jgz, z))/self.tau
		self.x += (dxdt*dt).flatten()
		self.r = np.tanh(self.x)
		# print(self.r)

		return np.dot(self.w, self.r)


	def train(self, training_time, dt, f):
		''' train network to learn function f '''
		z = f(0)

		self.x = np.random.randn(self.N_neurons)*self.sigma_w
		self.r = np.tanh(self.x)

		total_eminus = 0
		total_eplus = 0

		# self.P = np.identity(self.N_neurons)/self.alpha
		for t in range(-1, 10):
			z = self.step(f(t), dt)

		for t in range(10, training_time+1):
			# advance one step
			z = self.step(z, dt)

			# --- RLS algorithm ---
			eminus = np.dot(self.w, self.r) - f(t)

			# update matrix P
			denom = (1 + np.dot(np.transpose(self.r), np.dot(self.P, self.r)))
			dPdt = -1*np.outer(np.dot(self.P, self.r), np.dot(np.transpose(self.r), self.P)) / denom
			self.P += dPdt*dt

			# update weights
			dwdt = -1*eminus*np.dot(self.P, self.r)
			self.w += dwdt*dt

			eplus = np.dot(self.w, self.r) - f(t)

			total_eminus += abs(eminus)
			total_eplus += abs(eplus)

			# training feedback
			if t%12000 == 0:
				print(f"--- Step {t} ---")
				print(f"Avg e_minus = {total_eminus/12000}")
				print(f"Avg e_plus = {total_eplus/12000}")
				print(f"Diff = {(total_eminus-total_eplus)/12000}\n")

				total_eminus = 0
				total_eplus = 0