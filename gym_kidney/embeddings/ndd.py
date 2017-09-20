from gym_kidney import embeddings
from gym import spaces

import numpy as np
import networkx as nx

#
# Embeds number of non-directed donors.
#

class NddEmbedding(embeddings.Embedding):

	observation_space = spaces.Box(0, np.inf, (1,))

	def embed(self, G):
		ndd = 0
		for u in G.nodes_iter():
			if G.node[u]["ndd"]:
				ndd += 1
		return [ndd]
