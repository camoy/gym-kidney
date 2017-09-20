from embedding import Embedding
import numpy as np
import networkx as nx

#
# Embeds number of non-directed donors.
#

class NddEmbedding(Embedding):

	observation_space = spaces.Box(0, np.inf, (1,))

	def embed(self, G):
		ndd = 0
		for u in G.nodes_iter():
			if G.node[u]["ndd"]:
				ndd += 1
		return [ndd]
