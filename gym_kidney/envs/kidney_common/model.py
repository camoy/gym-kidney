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

	def _inv_map_2(self, d):
		"""
		Given dictionary d with possible non-unique values.
		Returns inverse dictionary.
		"""
		inv_map = {}
		for k, v in d.items():
			inv_map[v] = inv_map.get(v, [])
			inv_map[v].append(k)
		return inv_map

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

	def _depart(self, g, n):
		"""
		Given graph g, and number n. Removes n random vertices
		from g. Returns g.
		"""
		if n == 0: return False, g
		leave = self.rng.choice(g.nodes(), n, replace = False).tolist()
		g.remove_nodes_from(leave)
		g = nx.convert_node_labels_to_integers(g)
		return True, g

class ContrivedModel(_MixinModel):
	def __init__(self, rng):
		self.rng = rng
		self.log = []

	def reset(self):
		"""
		Returns contrived graph at initial state.
		"""
		g = nx.DiGraph([(0, 1)])
		nx.set_node_attributes(g, "ndd", { 0 : True, 1 : False })
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
				g.add_nodes_from([2, 3], ndd = False)
				g.add_edge(2, 3)
				g.add_edge(1, 2)
			# empty graph
			else:
				g = nx.DiGraph([(0, 1)])
				nx.set_node_attributes(g, "ndd", False)

			return g
		# reset on odd ticks
		else:
			return self.reset()

class HomogeneousModel(_MixinModel):
	def __init__(self, rng, rate, d, p_t, p_a):
		self.rng = rng
		self.rate = rate
		self.d = d
		self.p_t = p_t
		self.p_a = p_a
		self.log = [rate, d, p_t, p_a]

		self.n = rate / p_t
		self.p = d / rate
		self.p_d = p_t / rate

	def _arrive(self, g, n, p, p_a):
		"""
		Given graph g, number n, edge probability p, and NDD probability
		p_a. Adds n new vertices to graph (NDD with probability p_a)
		with edge probability p. Returns g.
		"""
		if n == 0: return False, g

		n0 = g.order()
		new = list(range(n0, n0+n))
		g.add_nodes_from(new, ndd = False)
		attr_ndd = nx.get_node_attributes(g, "ndd")

		for u in new:
			# NDD
			ualt = self.rng.rand() < p_a
			if ualt:
				s = { u : True }
				nx.set_node_attributes(g, "ndd", s)
				attr_ndd[u] = True

		# edges
		for u in new:
			ualt = attr_ndd[u]
			for v in g.nodes():
				if u == v: continue
				valt = attr_ndd[v]
				if self.rng.rand() < p and not valt:
					g.add_edge(u, v)
				if self.rng.rand() < p and not ualt:
					g.add_edge(v, u)

		return True, g

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
		a_n = 1
		a_p = self.p_t

		a = self.rng.binomial(a_n, a_p)
		changed1, g = self._arrive(g, a, self.p, self.p_a)

		# departure
		d_n = len(g.nodes())
		d = self.rng.binomial(d_n, self.p_d)
		changed2, g = self._depart(g, d)

		return changed1 or changed2, g

class HeterogeneousModel(_MixinModel):
	def __init__(self, rng, rate, d_l, d_h, p_s, p_t, p_a):
		self.rng = rng
		self.rate = rate
		self.d_l = d_l
		self.d_h = d_h
		self.p_s = p_s
		self.p_t = p_t
		self.p_a = p_a
		self.log = [rate, d_l, d_h, p_s, p_t, p_a]

		self.n = rate / p_t
		self.p_l = d_l / rate
		self.p_h = d_h / rate
		self.p_d = p_t / rate

	def _arrive(self, g, n, p, p_l, p_h, p_a):
		"""
		Given graph g, number n, high PRA probability p, low PRA edge
		probability p_l, high PRA edge probability p_h, and NDD
		probability p_a. Adds n new vertices to graph (NDD with
		probability p_a, high PRA with probability p) with edge
		probability p_l or p_h depending other vertices' PRA. Returns
		g.
		"""
		if n == 0: return False, g

		n0 = g.order()
		new = list(range(n0, n0+n))
		g.add_nodes_from(new, ndd = False)
		attr_ndd = nx.get_node_attributes(g, "ndd")
		attr_pra = nx.get_node_attributes(g, "pra")

		for u in new:
			# NDD
			ualt = self.rng.rand() < p_a
			if ualt:
				s = { u : True }
				nx.set_node_attributes(g, "ndd", s)
				attr_ndd[u] = True

			# PRA
			uhigh = self.rng.rand() < p
			val = "high" if uhigh else "low"
			s = { u : val }
			nx.set_node_attributes(g, "pra", "high")
			attr_pra[u] = val

		# edges
		for u in new:
			ualt = attr_ndd[u]
			uhigh = attr_pra[u] == "high"
			for v in g.nodes():
				if u == v: continue
				valt = attr_ndd[v]
				vhigh = attr_pra[v] == "high"
				p_v = p_h if vhigh else p_l
				p_u = p_h if uhigh else p_l
				if self.rng.rand() < p_v and not valt:
					g.add_edge(u, v)
				if self.rng.rand() < p_u and not ualt:
					g.add_edge(v, u)

		return True, g

	def reset(self):
		"""
		Returns heterogeneous graph at initial state (empty graph).
		"""
		return nx.DiGraph()

	def evolve(self, g, m, i):
		"""
		Given graph g, matching m, and tick i. Evolves heterogeneous
		graph. Returns g.
		"""

		# match
		g = self._process_matches(g, m)

		# arrival
		a_n = 1
		a_p = self.p_t

		a = self.rng.binomial(a_n, a_p)
		changed1, g = self._arrive(
			g,
			a,
			self.p_s,
			self.p_l,
			self.p_h,
			self.p_a)

		# departure
		d_n = len(g.nodes())
		d = self.rng.binomial(d_n, self.p_d)
		changed2, g = self._depart(g, d)

		return changed1 or changed2, g

class KidneyModel(_MixinModel):
	def __init__(self, rng, rate, p_t, data):
		self.rng = rng
		self.rate = rate
		self.p_t = p_t
		self.log = [rate, p_t]
		self.label_map = {}

		self.n = rate / p_t
		self.p_d = p_t / rate

		adj = np.loadtxt(data, delimiter = ",")
		self.glob = nx.DiGraph()
		self.glob = nx.from_numpy_matrix(adj, create_using = self.glob)

	def _arrive(self, g, n):
		"""
		Given graph g, number n. Adds n new vertices to graph according
		to compatibility matrix. Returns g.
		"""
		if n == 0: return False, g

		glob, lm = self.glob, self.label_map
		nodes = glob.nodes()
		ns = self.rng.choice(nodes, n, replace = True).tolist()

		# add vertices
		n0 = g.order()
		new = list(range(n0, n0+n))
		for i, u in enumerate(new):
			g.add_node(u, ndd = False)
			lm[u] = ns[i]

		# add out edges
		lmi = self._inv_map_2(lm)
		for u in new:
			gid = lm[u]
			for vs in list(map(lmi.get, glob.successors(gid))):
				if vs == None: continue
				for v in vs:
					if v in g.nodes():
						g.add_edge(u, v)

		# add in edges
		for u in new:
			gid = lm[u]
			for v in list(map(lmi.get, glob.predecessors(gid))):
				if vs == None: continue
				for v in vs:
					if v in g.nodes():
						g.add_edge(v, u)

		return True, g

	def reset(self):
		"""
		Returns kidney graph at initial state (empty graph).
		"""
		return nx.DiGraph()

	def evolve(self, g, m, i):
		"""
		Given graph g, matching m, and tick i. Evolves kidney
		graph. Returns g.
		"""

		# match
		g = self._process_matches(g, m)

		# arrival
		a_n = 1
		a_p = self.p_t

		a = self.rng.binomial(a_n, a_p)
		changed1, g = self._arrive(g, a)

		# departure
		d_n = len(g.nodes())
		d = self.rng.binomial(d_n, self.p_d)
		changed2, g = self._depart(g, d)

		return changed1 or changed2, g

