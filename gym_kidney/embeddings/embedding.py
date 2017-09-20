from gym import spaces

class Embedding:

	# params :: Dictionary
	params = {}

	# stats :: Dictionary
	stats = {}

	# observation_space :: Gym.Space
	observation_space = spaces.Box(0, 0, (0,))
	
	# embed :: NetworkX.Graph -> Array
	def embed(self, g):
		raise NotImplementedError
