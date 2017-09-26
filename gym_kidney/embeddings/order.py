from gym_kidney import embeddings
from gym import spaces

import numpy as np
import networkx as nx

#
# OrderEmbedding embeds the order of the graph.
#
class OrderEmbedding(embeddings.Embedding):

	observation_space = spaces.Box(0, np.inf, (1,))

	def embed(self, G):
		return np.array([G.order()])
