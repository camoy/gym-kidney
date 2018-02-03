from gym_kidney import models

import networkx as nx

class AshlagiModel(models.Model):

	def __init__(self, n, rho, p_h, p_l):
		self.n = n
		self.rho = rho 
		self.p_h = p_h
		self.p_l = p_l

		self.params = {
			"n": n,
			"rho": rho,
			"p_h": p_h,
			"p_l": p_l
		}

		self.stats = {}

	def arrive(self, G, rng):
		#n1 = G.order()
		#n2 = rng.poisson(self.m / self.k)
		#new = range(n1, n1 + n2)

		#for u in new:
		#ndd_u = rng.rand() < self.p_a
		high_u = rng.rand() < self.rho
		G.add_node(u, high = high_u)

		for v in G.nodes():
			#ndd_v = G.node[v]["ndd"]
			high_v = G.node[v]["high"]
			p_u = self.p_h if high_u else self.p_l
			p_v = self.p_h if high_v else self.p_l

			if u == v: continue
			if rng.rand() < p_v:
				G.add_edge(u, v)
			if rng.rand() < p_u:
				G.add_edge(v, u)

		return G

	# NOP
	def depart(self, G, rng):
		#n1 = G.order()
		#n2 = rng.binomial(n1, 1.0 / self.k)
		#old = rng.choice(G.nodes(), n2, replace = False).tolist()

		#G.remove_nodes_from(old)
		#self.stats["departed"] += n2
		#return nx.convert_node_labels_to_integers(G)
		return G

	def done(self, tick):
		return tick >= self.n
