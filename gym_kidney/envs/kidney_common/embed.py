import math
import networkx as nx
import numpy as np
import scipy.sparse as sp

#
# MAIN FUNCTIONS
#

def _walks(w, p0, tau):
	"""
	Given transition matrix w, initial distribution
	p0, and walk cap tau. Returns list ps, where
	ps[t] is the distribution over vertices after
	t steps.
	"""
	ps = [p0]
	for i in range(1, tau+1):
		ps += [w * ps[i-1]]
	return ps

def _degrees_inv(g):
	"""
	Given graph g. Returns inverse of degree matrix
	where 0 entries are ignored.
	"""
	n, degs = g.order(), g.degree().values()
	degs = list(map(lambda x : 0 if x == 0 else 1/x, degs))
	return sp.diags(degs, format = "csc")

def _trans(g, deg):
	"""
	Given graph g, and inverse degree matrix. Returns
	transition matrix for g (uniform over adjacent vertices).
	"""
	adj = nx.to_scipy_sparse_matrix(g, format = "csc")
	return (deg * adj).T

def _feature(g, p0, tau):
	"""
	Given graph g, initial distribution p0, and walk
	cap tau. Returns random walk feature vector (size
	dependent only on tau).
	"""
	n, m = g.order(), np.zeros((tau+1, tau+1))
	deg_inv = _degrees_inv(g)
	ps = _walks(_trans(g, deg_inv), p0, tau)
	dsqrt = deg_inv.sqrt()
	for s in range(tau):
		for t in range(s+1, tau+1):
			m[s, t] = sp.linalg.norm(dsqrt*ps[s] - dsqrt*ps[t])
	return m[np.triu_indices(tau+1, 1)].tolist()

#
# INITIAL DISTRIBUTIONS HELPERS
#

def _best_vertex(g, f):
	"""
	Given graph g and function f. Returns index of
	the vertex whose degree is closest to f(v)
	where v is the list of degrees.
	"""
	degs = g.degree()
	pr = nx.pagerank(g)
	k, v = list(degs.keys()), list(degs.values())
	cf = f(v)
	best_val = min(v, key = lambda x: abs(x-cf))
	cand = {}
	for k, v in degs.items():
		if v == best_val:
			cand[k] = pr[k]
	return max(cand, key = cand.get)

def _p0_dirac(g, v):
	"""
	Given graph g and vertex v. Returns Dirac
	delta distribution at vertex v.
	"""
	n = g.order()
	p0 = [0] * n
	p0[v] = 1
	return sp.csc_matrix(p0).T

#
# INITIAL DISTRIBUTIONS
#

def p0_min(g):
	"""
	Given graph g. Returns Dirac delta distribution
	at vertex with minimum degree.
	"""
	return _p0_dirac(g, _best_vertex(g, min))

def p0_max(g):
	"""
	Given graph g. Returns Dirac delta distribution
	at vertex with maximum degree.
	"""
	return _p0_dirac(g, _best_vertex(g, max))

def p0_median(g):
	"""
	Given graph g. Returns Dirac delta distribution
	at vertex with median degree.
	"""
	return _p0_dirac(g, _best_vertex(g, np.median))

def p0_mean(g):
	"""
	Given graph g. Returns Dirac delta distribution
	at vertex with mean degree.
	"""
	return _p0_dirac(g, _best_vertex(g, np.mean))

#
# WALK2VEC
# 

def embed(g, p0s, tau):
	"""
	Given graph g, list of initial distribution generating
	functions p0s, and walk cap tau. Returns embedding of g.
	"""

	# empty graph
	if g.order() == 0:
		return [0]*(int(len(p0s)*((tau**2+tau)/2)))

	# non-empty graphs
	phi = []
	for i, p0_i in enumerate(p0s):
		phi += _feature(g, p0_i(g), tau)
	return phi
