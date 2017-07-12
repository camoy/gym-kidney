import networkx as nx
import numpy as np
from . import convert as kc
import random

def inv_map(d):
	"""
	Given dictionary d. Returns inverse dictionary.
	"""
	return dict((v, k) for k, v in d.items())

def process_matches(g, m):
	"""
	Given graph g, and tuple of match structures m.
	Removes vertices in match from g. Returns g.
	"""
	_, _, n_map, ndd_map = kc.relabel(g)
	n_map, ndd_map = inv_map(n_map), inv_map(ndd_map)

	# construct remove list
	cycle, chain = m
	remove = []
	for vs in cycle:
		remove += list(map(lambda u: n_map[u.id], vs))
	for c in chain:
		remove += [ndd_map[c.ndd_index]]
		remove += list(map(lambda u: n_map[u], c.vtx_indices))

	# remove
	g.remove_nodes_from(remove)

	return g

def reset(n, p):
	"""
	Returns contrived graph at initial state.
	"""
	g = nx.DiGraph([(0, 1)])
	nx.set_node_attributes(g, "altruist", { 0 : True, 1 : False })
	return g

def evolve(g, m, a, d, p, i):
	"""
	Given graph g, matching m, arrival rate a, death rate
	d, density p, and tick i. Evolves contrived graph based
	on parity. Returns g.
	"""
	# evolve on even ticks
	if i % 2 == 0:
		g = process_matches(g, m)

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
		return reset(0, 0)
