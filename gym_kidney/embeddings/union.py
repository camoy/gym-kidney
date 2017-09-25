from gym_kidney import embeddings
from gym import spaces

import numpy as np
import networkx as nx

#
# Unions set of embeddings.
#

class UnionEmbedding(embeddings.Embedding):

	def __init__(self, embeddings):
		len = len(embeddings)
		low, high = np.inf, -np.inf

		for e in self.embeddings:
			low = min(low, e.observation_space.low)
			high = max(high, e.observation_space.high)
			self.params = {**self.params, **e.params}

		self.observation_space = spaces.Box(low, high, (len,))
		self.embeddings = embeddings

	def embed(self, G):
		embedding = np.array([])

		for e in self.embeddings:
			embedding = np.concatenate((embedding, e.embed(G)))

		return embedding
