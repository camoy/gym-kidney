from gym_kidney import embeddings
from gym import spaces

import numpy as np
import networkx as nx

#
# NOP Embedding.
#

class NopEmbedding(embeddings.Embedding):

	observation_space = spaces.Box(0, 0, (0,))

	def embed(self, G):
		return []
