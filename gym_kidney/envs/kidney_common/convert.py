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
	for e in g.edges():
		u, v = e
		if not attrs[u]:
			digraph.add_edge(
				1.0,
				digraph.vs[n_map[u]],
				digraph.vs[n_map[v]])

	# ndds
	ndds = [ks.kidney_ndds.Ndd() for _ in range(n_ndds)]
	for e in g.edges():
		u, v = e
		if attrs[u]:
			edge = ks.kidney_ndds.NddEdge(digraph.vs[n_map[v]], 1.0)
			ndds[ndd_map[u]].add_edge(edge)
	
	return digraph, ndds
