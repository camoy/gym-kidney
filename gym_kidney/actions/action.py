from gym import spaces
from gym_kidney import _solver

import networkx as nx

class Action:

	# params :: Dictionary
	params = {}

	# stats :: Dictionary
	stats = {}

	# action_space :: Gym.Space
	action_space = spaces.Discrete(2)

	# do_action :: ... -> (NetworkX.Graph, Reward)
	def do_action(G, action):
		raise NotImplementedError

	# _relabel :: NetworkX.Graph -> (Int, Int, DD Dict, NDD Dict)
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

		return n_dd, n_ndd, d_dd, d_ndd

	# _inv_dict :: Dictionary -> Dictionary
	def _inv_dict(self, d):
		return dict((v, k) for k, v in d.items())

	# _process_matches :: NetworkX.Graph -> Matching -> NetworkX.Graph
	def _process_matches(self, G, M):
		if len(M) == 0:
			return G

		_, _, d_dd, d_ndd = self._relabel(G)
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

	# _nx_to_ks :: NetworkX.Graph -> (KS.Digraph, KS.NDDs)
	def _nx_to_ks(self, G):
		n_dd, n_ndd, d_dd, d_ndd = self._relabel(G)

		dd = _solver.Digraph(n_dd)
		for u, v, d in G.edges(data = True):
			if not G.node[u]["ndd"]:
				dd.add_edge(
					d["weight"] if ("weight" in d) else 1.0,
					dd.vs[d_dd[u]],
					dd.vs[d_dd[v]])

		ndd = [_solver.kidney_ndds.Ndd() for _ in range(n_ndd)]
		for u, v, d in G.edges(data = True):
			if G.node[u]["ndd"]:
				edge = _solver.kidney_ndds.NddEdge(
					dd.vs[d_dd[v]],
					d["weight"] if ("weight" in d) else 1.0)
				ndd[d_ndd[u]].add_edge(edge)
		
		return dd, ndd
