from embedding import Embedding
import numpy as np
import networkx as nx

#
# Embeds order of the graph.
#

class OrderEmbedding(Embedding):

	observation_space = spaces.Box(0, np.inf, (1,))

	def embed(self, G):
		return [G.order()]
