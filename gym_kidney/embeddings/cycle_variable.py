from gym_kidney import embeddings
from gym import spaces

import numpy as np
import scipy.special as sp
import networkx as nx

#
# CycleVariableEmbedding embeds an estimate for the number of cycles in the
# graph using a variable number of samples.
# - successes : Nat, number of successes before stopping
# - sample_cap : Nat, maximum samples to take before quitting
# - cycle_length : Nat, cycle length under consideration
#
class CycleVariableEmbedding(embeddings.Embedding):

	observation_space = spaces.Box(0, np.inf, (1,))

	def __init__(self, successes, sample_cap, cycle_length):
		self.successes = successes
		self.sample_cap = sample_cap
		self.cycle_length = cycle_length

	def embed(self, G, rng):
		if G.order() < self.cycle_length:
			return np.array([0.0], dtype = "f")

		succ, samples = 0, 0
		max_cycle = sp.binom(G.order(), self.cycle_length)

		for _ in range(self.sample_cap):
			us = rng.choice(G.nodes(), self.cycle_length)
			H = G.subgraph(us.tolist())
			samples += 1

			for c in nx.simple_cycles(H):
				if len(c) == self.cycle_length:
					succ += 1
					break
			if succ >= self.successes:
				break
			
		return np.array([max_cycle * (succ / samples)], dtype = "f")
