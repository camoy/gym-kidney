import math
import networkx as nx
import numpy as np
import scipy.sparse as sp

#
# MAIN FUNCTIONS
#

def _alpha(g, alpha):
	"""
	Given graph g and vertex v. Returns Dirac
	delta distribution at vertex v.
	"""
	n = g.order()
	v = [alpha/n] * n
	return sp.csc_matrix(v).T

def _walks(g, w, p0, tau, alpha):
	"""
	Given graph g, transition matrix w, initial distribution
	p0, jump probability alpha, and walk cap tau. Returns
	list ps, where ps[t] is the distribution over vertices
	after t steps.
	"""
	ps = [p0]
	n = g.order()
	for i in range(1, tau+1):
		ps += [_alpha(g, alpha) + (1-alpha)*w*ps[i-1]]
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

def _feature(g, p0, tau, alpha):
	"""
	Given graph g, initial distribution p0, jump probability
	alpha, and walk cap tau. Returns random walk feature vector
	(size dependent only on tau).
	"""
	n, m = g.order(), []
	deg_inv = _degrees_inv(g)
	ps = _walks(g, _trans(g, deg_inv), p0, tau, alpha)
	for s in range(tau):
		for t in range(s+1, tau+1):
			m += [sp.linalg.norm(ps[s] - ps[t])]
	return m

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
	k, v = list(degs.keys()), list(degs.values())
	cf = f(v)
	best_val = min(v, key = lambda x: abs(x-cf))
	return k[v.index(best_val)]

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

def embed(g, p0s, tau, alpha):
	"""
	Given graph g, list of initial distribution generating
	functions p0s, jump probability alpha,and walk cap tau.
	Returns embedding of g.
	"""

	# empty graph
	if g.order() == 0:
		return [0]*(int(len(p0s)*((tau**2+tau)/2)))

	# non-empty graphs
	phi = []
	for i, p0_i in enumerate(p0s):
		phi += _feature(g, p0_i(g), tau, alpha)
	return phi
