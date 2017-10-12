from gym_kidney import embeddings
from gym import spaces

import numpy as np
import scipy.special as sp
import networkx as nx

#
# CycleFixedEmbedding embeds an estimate for the number of cycles in the graph
# using a fixed number of samples.
# - sample_size : Nat, number of samples to take
# - cycle_length : Nat, cycle length under consideration
#
class CycleFixedEmbedding(embeddings.Embedding):

	observation_space = spaces.Box(0, np.inf, (1,))

	def __init__(self, sample_size, cycle_length):
		self.sample_size = sample_size
		self.cycle_length = cycle_length

	def embed(self, G, rng):
		if G.order() < self.cycle_length:
			return np.array([0.0], dtype = "f")

		succ = 0
		max_cycle = sp.binom(G.order(), self.cycle_length)

		for _ in range(self.sample_size):
			us = rng.choice(G.nodes(), self.cycle_length)
			H = G.subgraph(us.tolist())

			for c in nx.simple_cycles(H):
				if len(c) == self.cycle_length:
					succ += 1
					break

		val = [max_cycle * (succ / self.sample_size)]
		return np.array(v, dtype = "f")
