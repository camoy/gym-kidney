from gym_kidney import embeddings
from gym import spaces

import numpy as np
import networkx as nx

#
# UnionEmbedding unions a set of embeddings.
#
class UnionEmbedding(embeddings.Embedding):

	def __init__(self, embeddings):
		n, low, high = 0, np.inf, -np.inf
		names = []

		for e in embeddings:
			sp = e.observation_space
			n += 1 if sp.shape == () else sp.shape[0]
			low = min(low, e.observation_space.low[0])
			high = max(high, e.observation_space.high[0])
			self.params = {**self.params, **e.params}
			names += [type(e).__name__]

		self.observation_space = spaces.Box(low, high, (n,))
		self.embeddings = embeddings
		self.params = { **self.params, "embeddings" : ";".join(names) }

	def embed(self, G, rng):
		embedding = np.array([], dtype = "f")

		for e in self.embeddings:
			embedding = np.concatenate((embedding, e.embed(G, rng)))

		return embedding
