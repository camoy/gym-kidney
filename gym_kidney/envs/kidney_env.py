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
		# parameters
		self.tau = 7
		self.n = 64
		self.density = 0.05
		self.arrival = 0.66
		self.death = 0.05
		self.cycle_cap = 3
		self.chain_cap = 3
		self.seed = None
		self.episode_len = 100
		self.init_distrs = [kc.p0_max, kc.p0_mean]

		# seeds
		# random.seed(self.seed)
		# np.random.seed(self.seed)

		# spaces
		obs_size = len(self.init_distrs)*int((self.tau**2 + self.tau)/2)
		self.action_space = spaces.Discrete(2)
		self.observation_space = spaces.Box(
			-np.inf,
			np.inf,
			(obs_size,))

		# reset
		self._reset()

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
		self.graph = kc.evolve(
			self.graph,
			match,
			self.arrival,
			self.death,
			self.density,
			self.tick)
		self.tick = self.tick + 1

		# return values
		embedding = self._get_obs()
		done = self.tick >= self.episode_len
		return embedding, reward, done, {}

	def _reset(self):
		# state
		self.tick = 0
		self.graph = kc.reset(
			self.n,
			self.density)
		return self._get_obs()

	def _get_obs(self):
		# embed
		return kc.embed(self.graph, [kc.p0_max, kc.p0_mean], self.tau)

	def _render(self, mode = "human", close = False):
		if close:
			return
		if self.tick == 0:
			plt.ion()

		g = self.graph
		attrs = nx.get_node_attributes(g, "altruist")
		values = ["red" if attrs[v] else "blue" for v in g.nodes()]

		plt.clf()
		nx.draw(g,
			pos = nx.circular_layout(g),
			node_color = values)
		plt.pause(0.01)
		return []
