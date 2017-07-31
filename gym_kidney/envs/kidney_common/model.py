import networkx as nx
import numpy as np
from . import convert as kc
import math
import csv

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

		# log matched
		for v in remove:
			self._log("match", g.node[v])

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

		# log departed
		for v in leave:
			self._log("depart", g.node[v])

		g.remove_nodes_from(leave)
		g = nx.convert_node_labels_to_integers(g)
		return True, g

	def _log(self, type, v):
		if not self.logd: return

		key = "%s_%s_%s" % (type, v["b1"], v["b2"])
		self.logd[key] += 1

	def reset_log(self):
		ty = ["arrive", "depart", "match"]
		dbl = pbl  = ["-", "A", "B", "AB", "O"]
		co = [(a, b, c) for a in ty for b in dbl for c in pbl]
		for k in co:
			self.logd["%s_%s_%s" % k] = 0

	def evolve(self, g, m, i):
		"""
		Given graph g, matching m, and tick i. Evolves graph.
		Returns g.
		"""

		# match
		g = self._process_matches(g, m)

		# arrival
		a_n = math.ceil(self.m / (self.k-1.0))
		a_p = (self.k-1.0) / self.k

		if a_n == 1:
			a_p = self.m / self.k

		a = self.rng.binomial(a_n, a_p)
		changed1, g = self._arrive(g, a)

		# departure
		d_n = len(g.nodes())
		d = self.rng.binomial(d_n, 1.0 / self.k)
		changed2, g = self._depart(g, d)

		return changed1 or changed2, g

class ContrivedModel(_MixinModel):
	def __init__(self, rng):
		self.rng = rng
		self.k = 10
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

			return True, g
		# reset on odd ticks
		else:
			return True, self.reset()

class HomogeneousModel(_MixinModel):
	def __init__(self, rng,  m, k, d, p_a):
		# parameters
		self.rng = rng
		self.m = float(m)
		self.k = float(k)
		self.d = float(d)
		self.p_a = float(p_a)

		# calculated
		self.log = [m, k, d, p_a]
		self.p = d / m

	def _arrive(self, g, n):
		"""
		Given graph g, number n, edge probability p, and NDD probability
		p_a. Adds n new vertices to graph (NDD with probability p_a)
		with edge probability p. Returns g.
		"""
		if n == 0: return False, g

		p, p_a = self.p, self.p_a
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

class HeterogeneousModel(_MixinModel):
	def __init__(self, rng,  m, k, d_l, d_h, p_s, p_a):
		# parameters
		self.rng = rng
		self.m = float(m)
		self.k = float(k)
		self.d_l = float(d_l)
		self.d_h = float(d_h)
		self.p_s = float(p_s)
		self.p_a = float(p_a)

		# calculated
		self.log = [m, d_l, d_h, p_s, p_a]
		self.p_l = d_l / m
		self.p_h = d_h / m

	def _arrive(self, g, n):
		"""
		Given graph g, number n, high PRA probability p, low PRA edge
		probability p_l, high PRA edge probability p_h, and NDD
		probability p_a. Adds n new vertices to graph (NDD with
		probability p_a, high PRA with probability p) with edge
		probability p_l or p_h depending other vertices' PRA. Returns
		g.
		"""
		if n == 0: return False, g

		p, p_l, p_h, p_a = self.p, self.p_l, self.p_h, self.p_a
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


class KidneyModel(_MixinModel):
	def __init__(self, rng, m, k, data, details):
		# parameters
		self.rng = rng
		self.m = float(m)
		self.k = float(k)

		# calculated
		self.logd = {}
		self.log = {"m": m, "k": k}

		# log details
		self.reset_log()

		# adjacency matrix
		adj = np.loadtxt(data, delimiter = ",")
		self.glob = nx.DiGraph()
		self.glob = nx.from_numpy_matrix(adj, create_using = self.glob)

		# details
		with open(details, mode = "r") as handle:
			read = csv.reader(handle)
			for row in read:
				u = self.glob.node[int(row[0])]
				u["ndd"] = row[1] == "1"
				u["b1"] = row[2]
				u["b2"] = row[3]

	def _arrive(self, g, n):
		"""
		Given graph g, number n. Adds n new vertices to graph according
		to compatibility matrix. Returns g.
		"""
		if n == 0: return False, g

		glob = self.glob
		nodes = glob.nodes()

		# add vertices
		n0 = g.order()
		new = list(range(n0, n0+n))
		for i, u in enumerate(new):
			ulab = self.rng.randint(0, self.glob.order())
			v = self.glob.node[ulab]
			g.add_node(
				u,
				ndd = v["ndd"],
				b1 = v["b1"],
				b2 = v["b2"],
				gid = ulab)
			self._log("arrive", g.node[u])

		# labelings
		lm = nx.get_node_attributes(g, "gid")
		lmi = self._inv_map_2(lm)

		# add out edges
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
			for vs in list(map(lmi.get, glob.predecessors(gid))):
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
