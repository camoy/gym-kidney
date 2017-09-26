from gym_kidney import embeddings
from gym import spaces

import numpy as np
import networkx as nx

#
# ChainEmbedding embeds the sum of longest chains possible from
# all non-directed donors.
#
class ChainEmbedding(embeddings.Embedding):

	observation_space = spaces.Box(0, np.inf, (1,))

	def embed(self, G):
		len = 0
		paths = self._longest_paths(G)
		for u in G.nodes_iter():
			if G.node[u]["ndd"]:
				len += paths[u]
		return np.array([len])

	def _longest_paths(G):
		dist = {}

		for node in nx.topological_sort(G):
			pairs = [(dist[v][0] + 1, v) for v in G.pred[node]] 
			if pairs:
				dist[node] = max(pairs)
			else:
				dist[node] = (0, node)

		return dist
