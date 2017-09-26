from gym import spaces

#
# Embedding is an abstract class defining the interface every
# embedding method must conform to.
#
class Embedding:
	# params :: Dictionary
	# The parameters defining the embedding
	params = {}

	# stats :: Dictionary
	# The values to record after embedding
	stats = {}

	# observation_space :: Gym Space
	# The observation space of the gym
	observation_space = spaces.Box(0, 0, (0,))
	
	# embed :: NetworkX Graph -> NumPy Array
	# Embeds the graph into a fixed-size vector
	def embed(self, G):
		raise NotImplementedError
