from gym_kidney import embeddings
from gym import spaces

import numpy as np
import networkx as nx

#
# NormalizeEmbedding normalizes the entries of an embedding.
#
class NormalizeEmbedding(embeddings.Embedding):

	def __init__(self, embedding, multipliers):
		sp = embedding.observation_space

		low_mult = min(multipliers)
		high_mult = max(multipliers)
		low = sp.low[0] * low_mult
		high = sp.high[0] * high_mult
		n = sp.n if sp.shape == () else sp.shape[0]

		self.observation_space = spaces.Box(low, high, (n,))
		self.embedding = embedding
		self.multipliers = multipliers

	def embed(self, G, rng):
		embedding = self.embedding.embed(G, rng)

		for i,_ in enumerate(embedding):
			embedding[i] *= self.multipliers[i]

		return embedding
