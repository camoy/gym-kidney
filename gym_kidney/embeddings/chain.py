from gym_kidney import embeddings
from gym import spaces

import numpy as np
import networkx as nx

#
# ChainEmbedding embeds the sum of longest chains possible from
# all non-directed donors.
# - chain_length : Nat, chain length under consideration
#
class ChainEmbedding(embeddings.Embedding):

	observation_space = spaces.Box(0, np.inf, (1,))

	def __init__(self, chain_length):
		self.chain_length = chain_length

	def embed(self, G, rng):
		len = 0

		for u in G.nodes_iter():
			if G.node[u]["ndd"]:
				len += self._longest_path(G, u)

		return np.array([len], dtype = "f")

	def _longest_path(self, G, u):
		return len(nx.dag_longest_path(nx.bfs_tree(G, u)))
