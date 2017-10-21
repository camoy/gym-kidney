from gym_kidney import embeddings
from gym import spaces

import numpy as np
import networkx as nx

#
# OmniscientEmbedding embeds the number to depart in the next tick.
#
class OmniscientEmbedding(embeddings.Embedding):

	observation_space = spaces.Box(0, np.inf, (1,))
	next_depart = 0

	def depart_number(self, G, rng):
		n1 = G.order()
		last_depart = self.next_depart
		self.next_depart = rng.binomial(n1, 1.0 / self.env.model.k)
		return last_depart

	def embed(self, G, rng):
		return np.array([self.next_depart], dtype = "f")
