import networkx as nx
import numpy as np
from . import convert as kc
import math

class _MixinModel:
	def _inv_map(self, d):
		"""
		Given dictionary d. Returns inverse dictionary.
		"""
		return dict((v, k) for k, v in d.items())

	def _process_matches(self, g, m):
		"""
		Given graph g, and tuple of match structures m.
		Removes vertices in match from g. Returns g.
		"""
		if len(m) == 0: return g

		_, _, n_map, ndd_map = kc.relabel(g)
		n_map = self._inv_map(n_map)
		ndd_map = self._inv_map(ndd_map)

		# construct remove list
		cycle, chain = m
		remove = []
		for vs in cycle:
			remove += list(map(lambda u: n_map[u.id], vs))
		for c in chain:
			vtx = c.vtx_indices
			remove += [ndd_map[c.ndd_index]]
			remove += list(map(lambda u: n_map[u], vtx))

		# remove
		g.remove_nodes_from(remove)
		g = nx.convert_node_labels_to_integers(g)

		return g

	def _arrive(self, g, n, p, p_a):
		if n == 0: return g

		n0 = g.order()
		new = list(range(n0, n0+n))
		g.add_nodes_from(new, altruist = False)
		attrs = nx.get_node_attributes(g, "altruist")

		for u in new:
			ualt = self.rng.rand() < p_a
			if ualt:
				s = { u : True }
				nx.set_node_attributes(g, "altruist", s)
				attrs[u] = True

			for v in g.nodes():
				valt = attrs[v]
				if u == v: continue
				if self.rng.rand() < p and not valt:
					g.add_edge(u, v)
				if self.rng.rand() < p and not ualt:
					g.add_edge(v, u)

		return g

	def _depart(self, g, n):
		if n == 0: return g
		leave = self.rng.choice(g.nodes(), n, replace = False).tolist()
		g.remove_nodes_from(leave)
		g = nx.convert_node_labels_to_integers(g)
		return g

class ContrivedModel(_MixinModel):
	def __init__(self, rng):
		self.rng = rng
		self.log = []

	def reset(self):
		"""
		Returns contrived graph at initial state.
		"""
		g = nx.DiGraph([(0, 1)])
		nx.set_node_attributes(g, "altruist", { 0 : True, 1 : False })
		return g

	def evolve(self, g, m, i):
		"""
		Given graph g, matching m, and tick i. Evolves contrived
		graph based on parity. Returns g.
		"""

		# evolve on even ticks
		if i % 2 == 0:
			g = self._process_matches(g, m)

			# unmatched chain
			if g.has_node(1):
				g.add_nodes_from([2, 3], altruist = False)
				g.add_edge(2, 3)
				g.add_edge(1, 2)
			# empty graph
			else:
				g = nx.DiGraph([(0, 1)])
				nx.set_node_attributes(g, "altruist", False)

			return g
		# reset on odd ticks
		else:
			return self.reset()

class HomogeneousModel(_MixinModel):
	def __init__(self, rng, rate, k, p, p_a):
		self.rng = rng
		self.rate = rate
		self.k = k
		self.p = p
		self.p_a = p_a
		self.log = [rate, k, p, p_a]

	def reset(self):
		"""
		Returns homogeneous graph at initial state (empty graph).
		"""
		return nx.DiGraph()

	def evolve(self, g, m, i):
		"""
		Given graph g, matching m, and tick i. Evolves homogeneous
		graph based. Returns g.
		"""

		# match
		g = self._process_matches(g, m)

		# arrival
		a_n = math.floor(self.rate/(self.k-1) + 0.5)
		if a_n == 0:
			a_n = 1
			a_p = self.rate/self.k
		else:
			a_p = (self.k-1)/self.k

		a = self.rng.binomial(a_n, a_p)
		g = self._arrive(g, a, self.p, self.p_a)

		# departure
		d_n = len(g.nodes())
		d_p = 1/self.k
		d = self.rng.binomial(d_n, d_p)
		g = self._depart(g, d)

		return g
