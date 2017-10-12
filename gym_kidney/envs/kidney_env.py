import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import networkx as nx

#
# KidneyEnv is the main environment driver.
#
class KidneyEnv(gym.Env):
	metadata = { "render.modes" : ["human"] }

	def setup(self):
		self.action_space = self.action.action_space
		self.observation_space = self.embedding.observation_space
		self._seed()

	def _seed(self, seed = None):
		self.rng, seed = seeding.np_random(seed)
		self.rng_seed = seed
		return [seed]

	def _step(self, action):
		G = self.G
		G, reward = self.action.do_action(G, action)
		G, done = self.model.evolve(G, self.rng, self.tick)

		self.G = G
		self.tick += 1

		return self._obs(), reward, done, {}

	def _reset(self):
		self.logger.output_log(self)
		self.tick = 0
		self.G = nx.DiGraph()
		self.seed(self.rng_seed + 1)
		return self._obs()

	def _obs(self):
		return self.embedding.embed(self.G, self.rng)

	def _render(self, mode = "human", close = False):
		if close:
			return

		import matplotlib.pyplot as plt

		if self.tick == 0:
			plt.ion()

		G = self.G
		attrs = nx.get_node_attributes(G, "ndd")
		values = ["red" if attrs[v] else "blue" for v in G.nodes()]

		plt.clf()
		nx.draw(G,
			pos = nx.circular_layout(G),
			node_color = values)
		plt.pause(0.01)

		return []

