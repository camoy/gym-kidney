import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym_kidney.envs import kidney_solver as ks
from gym_kidney.envs import kidney_common as kc

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class KidneyEnv(gym.Env):
	metadata = { "render.modes" : ["human"] }

	def __init__(self):
		# default parameters
		self.tau = 5
		self.cycle_cap = 3
		self.chain_cap = 3
		self.eps_len = 200
		self.init_distrs = [kc.p0_max, kc.p0_mean]

		# spaces
		obs_size = len(self.init_distrs)*int((self.tau**2 + self.tau)/2)
		self.action_space = spaces.Discrete(2)
		self.observation_space = spaces.Box(
			-np.inf,
			np.inf,
			(obs_size,))

		# initialize
		self._seed()
		rate, k, p, pa = 25, 50, 0.05, 0.1
		self.model = kc.HomogeneousModel(self.rng, rate, k, p, pa)
		self._reset()

	def _seed(self, seed = None):
		self.seed = seed
		self.rng, seed = seeding.np_random(seed)
		return [seed]

	def _step(self, action):
		reward = 0
		match = ([], [])

		# match
		if action == 1:
			d, ndds = kc.nx_to_ks(self.graph)
			cfg = ks.kidney_ip.OptConfig(
				d,
				ndds,
				self.cycle_cap,
				self.chain_cap)
			soln = ks.solve_kep(cfg, "picef")
			match = (soln.cycles, soln.chains)
			reward = soln.total_score

		# evolve
		self.graph = self.model.evolve(
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
		self.graph = self.model.reset()
		return self._get_obs()

	def _get_obs(self):
		# embed
		return kc.embed(self.graph, self.init_distrs, self.tau)

	def _render(self, mode = "human", close = False):
		if close:
			return
		if self.tick == 0:
			plt.ion()

		# define colors
		g = self.graph
		attrs = nx.get_node_attributes(g, "ndd")
		values = ["red" if attrs[v] else "blue" for v in g.nodes()]

		# draw graph
		plt.clf()
		nx.draw(g,
			pos = nx.circular_layout(g),
			node_color = values)
		plt.pause(0.01)

		return []
