from embedding import Embedding
import numpy as np
import networkx as nx

#
# NOP Embedding.
#

class NopEmbedding(Embedding):

	observation_space = spaces.Box(0, 0, (0,))

	def embed(self, G):
		return []
