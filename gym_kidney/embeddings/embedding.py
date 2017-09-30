from gym import spaces

#
# Embedding is an abstract class defining the interface every
# embedding method must conform to.
#
class Embedding:
	# params : Dict
	# The parameters defining the embedding
	params = {}

	# stats : Dict
	# The values to record after embedding
	stats = {}

	# observation_space : Space
	# The observation space of the gym
	observation_space = spaces.Box(0, 0, (0,))
	
	# embed : Graph, RNG -> NP Array
	# Embeds the graph into a fixed-size vector
	def embed(self, G, rng):
		raise NotImplementedError
