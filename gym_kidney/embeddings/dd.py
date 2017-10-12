from gym_kidney import embeddings
from gym import spaces

import numpy as np
import networkx as nx

#
# DdEmbedding embeds the number of directed donors.
#
class DdEmbedding(embeddings.Embedding):

	observation_space = spaces.Box(0, np.inf, (1,))

	def embed(self, G, rng):
		dd = 0
		for u in G.nodes_iter():
			if not G.node[u]["ndd"]:
				dd += 1
		return np.array([dd], dtype = "f")
