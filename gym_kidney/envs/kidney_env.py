import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym_kidney.envs import kidney_solver as ks
from gym_kidney.envs import kidney_common as kc

import numpy as np
import networkx as nx
import os
#import matplotlib.pyplot as plt

class KidneyEnv(gym.Env):
	metadata = { "render.modes" : ["human"] }

	def __init__(self):
		# default parameters
		self.atoms = 100
		self.tau = 7
		self.alpha = 0.05
		self.cycle_cap = 3
		self.chain_cap = 3
		self.t = 5
		self.d_path = None
		self.training = False
		self.dict = None
		self.lembed = []
		self.init_distrs = [kc.p0_max, kc.p0_mean]

		# initialize
		self._seed()
		self.model = kc.ContrivedModel(self.rng)
		self._setup()
		self._reset()

	def _setup(self):
		# spaces
		obs_size = self.atoms
		self.action_space = spaces.Box(
			-2,
			2,
			(1,))
		self.observation_space = spaces.Box(
			-np.inf,
			np.inf,
			(obs_size,))

		# embedding
		self.lembed = [0] * obs_size

		# length
		self.eps_len = self.model.k * self.t

		# seed
		self._seed(seed = self.seed)

		# dictionary
		if self.d_path and os.path.exists(self.d_path):
			self.dict = np.loadtxt(self.d_path)
			self.dict = np.asfortranarray(self.dict, dtype=float)
		else:
			self.dict = None

	def _seed(self, seed = None):
		self.seed = seed
		self.rng, seed = seeding.np_random(seed)
		return [seed]

	def _step(self, action):
		reward = 0
		match = ([], [])

		self.graph = kc.reweight(self.graph, action)
		d, ndds = kc.nx_to_ks(self.graph)
		cfg = ks.kidney_ip.OptConfig(
			d,
			ndds,
			self.cycle_cap,
			self.chain_cap)
		soln = ks.solve_kep(cfg, "picef")
		match = (soln.cycles, soln.chains)
		rew_cycles = sum(map(lambda x: len(x.vtx_indices), soln.cycles))
		rew_chains = sum(map(lambda x: len(x.vtx_indices), soln.chains))
		reward = rew_cycles + rew_chains

		# evolve
		self.changed, self.graph = self.model.evolve(
			self.graph,
			match,
			self.tick)
		self.tick = self.tick + 1

		# return values
		embedding = self._get_obs()
		done = self.tick >= self.eps_len
		return embedding, reward, done, {}

	def _reset(self):
		# state
		self.tick = 0
		self.changed = True
		self.graph = self.model.reset()

		# dictionary
		if not (self.dict is None) and self.training:
			np.savetxt(self.d_path, self.dict)

		return self._get_obs()

	def _get_obs(self):
		#if self.changed:
		graph = self.graph
		tau, alpha = self.tau, self.alpha

		# new embedding
		#self.changed = False
		if self.training:
			self.dict = kc.train(
				graph,
				tau,
				alpha,
				d = self.dict,
				params = { "K": self.atoms })
		#elif not (self.dict is None):
			self.lembed = kc.phi(
				graph,
				self.init_distrs,
				tau,
				alpha)#,
				#self.dict,
				#kc.pool_avg)

		# return cached embedding
		return np.array(self.lembed)

	def _render(self, mode = "human", close = False):
		if close:
			return
		#if self.tick == 0:
		#	plt.ion()

		# define colors
		g = self.graph
		attrs = nx.get_node_attributes(g, "ndd")
		values = ["red" if attrs[v] else "blue" for v in g.nodes()]

		# draw graph
		#plt.clf()
		#nx.draw(g,
		#	pos = nx.circular_layout(g),
		#	node_color = values)
		#plt.pause(0.01)

		return []
