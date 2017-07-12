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

def _degrees(g):
	"""
	Given graph g. Returns sparse diagonal matrix
	containing degree of vertices.
	"""
	n, degs = g.order(), g.degree()
	d = sp.lil_matrix((n, n))
	for i in range(n):
		d[i, i] = degs[i]
	return d.tocsc()

def _degrees_inv(g):
	"""
	Given graph g. Returns inverse of degree matrix
	where 0 entries are ignored.
	"""
	n = g.order()
	ret, degs = sp.lil_matrix((n, n)), _degrees(g)
	for i in range(n):
		d = degs[i, i]
		if d != 0:
			ret[i, i] = 1/d
	return ret.tocsc()

def _trans(g):
	"""
	Given graph g. Returns transition matrix for
	g (uniform over adjacent vertices).
	"""
	adj = nx.adjacency_matrix(g).tocsc()
	return (_degrees_inv(g) * adj).transpose()

def _feature(g, p0, tau):
	"""
	Given graph g, initial distribution p0, and walk
	cap tau. Returns random walk feature vector (size
	dependent only on tau).
	"""
	n, m = g.order(), np.zeros((tau+1, tau+1))
	ps = _walks(_trans(g), p0, tau)
	dsqrt = _degrees_inv(g).sqrt()
	for s in range(tau):
		for t in range(s+1, tau+1):
			m[s, t] = sp.linalg.norm(dsqrt*ps[s] - dsqrt*ps[t])
	return m[np.triu_indices(tau+1, 1)]

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
	best_val = min(v, key=lambda x:abs(x-cf))
	return k[v.index(best_val)]

def _p0_dirac_gen(g, f):
	"""
	Given graph g and function f. Returns Dirac
	delta distribution at vertex f(g).
	"""
	n, v = g.order(), f(g)
	p0 = np.zeros((n, n))
	p0[v, v] = 1
	return sp.csc_matrix(p0)

#
# INITIAL DISTRIBUTIONS
#

def p0_min(g):
	"""
	Given graph g. Returns Dirac delta distribution
	at vertex with minimum degree.
	"""
	return _p0_dirac_gen(g, (lambda g: _best_vertex(g, min)))

def p0_max(g):
	"""
	Given graph g. Returns Dirac delta distribution
	at vertex with maximum degree.
	"""
	return _p0_dirac_gen(g, (lambda g: _best_vertex(g, max)))

def p0_median(g):
	"""
	Given graph g. Returns Dirac delta distribution
	at vertex with median degree.
	"""
	return _p0_dirac_gen(g, (lambda g: _best_vertex(g, np.median)))

def p0_mean(g):
	"""
	Given graph g. Returns Dirac delta distribution
	at vertex with mean degree.
	"""
	return _p0_dirac_gen(g, (lambda g: _best_vertex(g, np.mean)))

def p0_dirac(g, i):
	"""
	Given graph g. Returns Dirac delta distribution
	at vertex i.
	"""
	return _p0_dirac_gen(g, (lambda g: i))

def p0_unif(g):
	"""
	Given graph g. Returns uniform distribution.
	"""
	n = g.order()
	return sp.csc_matrix((1/n)*np.identity(n))

#
# WALK2VEC VANILLA
# 

def embed(g, p0s, tau):
	"""
	Given graph g, list of initial distribution generating
	functions p0s, and walk cap tau. Returns embedding of g.
	"""
	phi = []
	for i, p0_i in enumerate(p0s):
		phi += [_feature(g, p0_i(g), tau)]
	return np.concatenate(phi)
