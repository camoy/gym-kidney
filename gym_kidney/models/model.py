class Model:

	# params :: Dictionary
	params = {}

	# stats :: Dictionary
	stats = {}

	# evolve :: ... -> (Boolean, NetworkX.Graph)
	def evolve(self, G, M, rng, tick):
		G = self._process_matches(G, M)
		G = self.arrive(G, rng)
		G = self.depart(G, rng)
		return self.done(tick), G

	# arrive :: NetworkX.Graph -> NetworkX.Graph
	def arrive(self, G):
		raise NotImplementedError

	# depart :: NetworkX.Graph -> NetworkX.Graph
	def depart(self, G):
		raise NotImplementedError

	# done :: Natural -> Boolean
	def done(self, tick):
		raise NotImplementedError

	# _relabel :: NetworkX.Graph -> (DD Dictionary, NDD Dictionary)
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
	def _inv_dict(self, d):
		return dict((v, k) for k, v in d.items())

	# _process_matches :: NetworkX.Graph -> Matching -> NetworkX.Graph
	def _process_matches(self, G, M):
		if len(M) == 0:
			return G

		d_dd, d_ndd = self._relabel(G)
		d_dd, d_ndd = self._inv_dict(d_dd), self._inv_dict(d_ndd)
		cycle, chain = M
		out = []

		for vs in cycle:
			out += list(map(lambda u: d_dd[u.id], vs))
		for c in chain:
			vs = c.vtx_indices
			out += [d_ndd[c.ndd_index]]
			out += list(map(lambda u: d_dd[u], vs))

		G.remove_nodes_from(out)
		return nx.convert_node_labels_to_integers(G)
