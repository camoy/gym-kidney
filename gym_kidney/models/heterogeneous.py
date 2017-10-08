from gym_kidney import models

import networkx as nx

#
# HeterogeneousModel evolves the graph according to a heterogeneous
# Erdős–Rényi random model.
# - m : Nat, expected vertices per period
# - k : Nat, ticks per period
# - p_l : [0, 1], probability of edge to patient with low PRA
# - p_h : [0, 1], probability of edge to patient with high PRA
# - p_a : [0, 1], probability of NDD
# - p_s : [0, 1], probability of patient with high PRA
# - len : Nat, ticks per episode
#
class HeterogeneousModel(models.Model):

	def __init__(self, m, k, p_l, p_h, p_a, p_s, len):
		self.m = m
		self.k = k
		self.p_l = p_l
		self.p_h = p_h
		self.p_a = p_a
		self.p_s = p_s
		self.len = len

		self.params = {
			"m": m,
			"k": k,
			"p_l": p_l,
			"p_h": p_h,
			"p_a": p_a,
			"p_s": p_s,
			"len": len
		}

		self.stats = {
			"arrived": 0,
			"departed": 0
		}

	def arrive(self, G, rng):
		n1 = G.order()
		n2 = rng.poisson(self.m / self.k)
		new = range(n1, n1 + n2)

		for u in new:
			ndd_u = rng.rand() < self.p_a
			high_u = rng.rand() < self.p_s
			G.add_node(u, ndd = ndd_u, high = high_u)

			for v in G.nodes():
				ndd_v = G.node[v]["ndd"]
				high_v = G.node[v]["high"]
				p_u = self.p_h if high_u else self.p_l
				p_v = self.p_h if high_v else self.p_l

				if u == v: continue
				if rng.rand() < p_v and not ndd_v:
					G.add_edge(u, v)
				if rng.rand() < p_u and not ndd_u:
					G.add_edge(v, u)

		self.stats["arrived"] += n2
		return G

	def depart(self, G, rng):
		n1 = G.order()
		n2 = rng.binomial(n1, 1.0 / self.k)
		old = rng.choice(G.nodes(), n2, replace = False).tolist()

		G.remove_nodes_from(old)
		self.stats["departed"] += n2
		return nx.convert_node_labels_to_integers(G)

	def done(self, tick):
		return tick >= self.len
