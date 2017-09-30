#
# Model is an abstract class to be implemented by all models
# generating the environment.
#
class Model:

	# params : Dict
	# The parameters defining the model 
	params = {}

	# stats : Dict
	# The values to record after evolving
	stats = {}

	# evolve : Graph, RNG, Nat -> (Graph,  Bool)
	# Evolves the graph by arriving and departing vertices
	def evolve(self, G, rng, tick):
		G = self.arrive(G, rng)
		G = self.depart(G, rng)
		return G, self.done(tick)

	# arrive : Graph -> Graph
	# Arrives vertices in the graph
	def arrive(self, G):
		raise NotImplementedError

	# depart : Graph -> Graph
	# Departs vertices in the graph
	def depart(self, G):
		raise NotImplementedError

	# done : Nat -> Bool
	# Whether the episode is over
	def done(self, tick):
		raise NotImplementedError
