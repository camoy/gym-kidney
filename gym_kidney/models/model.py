#
# Model is an abstract class to be implemented by all models
# generating the environment.
#
class Model:

	# params :: Dictionary
	# The parameters defining the model 
	params = {}

	# stats :: Dictionary
	# The values to record after evolving
	stats = {}

	# evolve :: NetworkX Graph, NumPy RNG, Nat -> (NetworkX Graph, Boolean)
	# Evolves the graph by arriving and departing vertices
	def evolve(self, G, rng, tick):
		G = self.arrive(G, rng)
		G = self.depart(G, rng)
		return G, self.done(tick)

	# arrive :: NetworkX Graph -> NetworkX Graph
	# Arrives vertices in the graph
	def arrive(self, G):
		raise NotImplementedError

	# depart :: NetworkX Graph -> NetworkX Graph
	# Departs vertices in the graph
	def depart(self, G):
		raise NotImplementedError

	# done :: Natural -> Boolean
	# Whether the episode is over
	def done(self, tick):
		raise NotImplementedError

	# _relabel :: NetworkX Graph -> (DD Dictionary, NDD Dictionary)
	# Returns relabeling dictionaries
	def _relabel(self, G):
		n_dd, n_ndd = 0, 0
		d_dd, d_ndd = {}, {}

		for u in G.nodes():
			if G.node[u]["ndd"]:
				d_ndd[u] = n_ndd
				n_ndd += 1
			else:
				d_dd[u] = n_dd
				n_dd += 1

		return d_dd, d_ndd

	# _inv_dict :: Dictionary -> Dictionary
	# Inverts dictionary
	def _inv_dict(self, d):
		return dict((v, k) for k, v in d.items())
