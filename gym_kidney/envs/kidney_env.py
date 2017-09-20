import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import networkx as nx

class KidneyEnv(gym.Env):

	def setup(self):
		self.action_space = self.action.action_space
		self.observation_space = self.embedding.observation_space
		self._seed()

	def _seed(self, seed = None):
		self.seed = seed
		self.rng, seed = seeding.np_random(seed)
		return [seed]

	def _step(self, action):
		G = self.G
		G, reward = self.action.do_action(G, action)
		G, done = self.model.evolve(G, self.tick, self.rng)

		self.G = G
		self.tick += 1

		return self._obs(), reward, done, {}

	def _reset(self):
		self.logger.output_log(self)
		self.tick = 0
		self.G = nx.DiGraph()
		return self._obs()

	def _obs(self):
		return self.embedding.embed(self.G)

	def _render(self, close = False):
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

