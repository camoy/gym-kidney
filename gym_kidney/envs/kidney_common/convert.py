import networkx as nx
from .. import kidney_solver as ks

def relabel(g):
	"""
	Given a NetworkX graph. Returns: number of donor-patient-pair
	vertices, altruistic donors, pair relabeling dictionary, and
	altruist relabeling dictionary.
	"""
	n, n_ndds = 0, 0
	n_map, ndd_map = {}, {}
	attrs = nx.get_node_attributes(g, "ndd")

	# separate altruists
	for u in g.nodes_iter():
		if attrs[u]:
			ndd_map[u] = n_ndds
			n_ndds += 1
		else:
			n_map[u] = n
			n += 1

	return n, n_ndds, n_map, ndd_map

def reweight(g, action):
	attrs = nx.get_node_attributes(g, "ndd")
	for u, v, d in g.edges(data = True):
		w = action[0] if attrs[u] or attrs[v] else 0
		d["weight"] = d["weight"] - 0.5*w
	return g

def nx_to_ks(g):
	"""
	Given a NetworkX graph. Returns a representation
	of the graph and NDDs suitable for the kidney
	solver.
	"""
	n, n_ndds, n_map, ndd_map = relabel(g)
	attrs = nx.get_node_attributes(g, "ndd")

	# digraph
	digraph = ks.Digraph(n)
	for u, v, d in g.edges(data = True):
		if not attrs[u]:
			digraph.add_edge(
				d["weight"],
				digraph.vs[n_map[u]],
				digraph.vs[n_map[v]])

	# ndds
	ndds = [ks.kidney_ndds.Ndd() for _ in range(n_ndds)]
	for u, v, d in g.edges(data = True):
		if attrs[u]:
			edge = ks.kidney_ndds.NddEdge(
				digraph.vs[n_map[v]],
				d["weight"])
			ndds[ndd_map[u]].add_edge(edge)
	
	return digraph, ndds
