import math
import networkx as nx
import numpy as np
import scipy.sparse as sp
import spams

#
# MAIN FUNCTIONS
#

def _alpha(g, alpha):
	"""
	Given graph g and vertex v. Returns Dirac
	delta distribution at vertex v.
	"""
	n = g.order()
	v = [alpha / float(n)] * n
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
		ps += [_alpha(g, alpha) + (1.0-alpha)*w*ps[i-1]]
	return ps

def _degrees_inv(g):
	"""
	Given graph g. Returns inverse of degree matrix
	where 0 entries are ignored.
	"""
	n, degs = g.order(), g.degree().values()
	degs = list(map(lambda x : 0 if x == 0 else 1.0/float(x), degs))
	return sp.diags(degs, format = "csc")

def _trans(g, deg):
	"""
	Given graph g, and inverse degree matrix. Returns
	transition matrix for g (uniform over adjacent vertices).
	"""
	adj = nx.to_scipy_sparse_matrix(g, format = "csc")
	return (deg * adj).T

def _dist(pr, ps, pt):
	num = pt.T.dot(np.divide(ps, pr)).item(0)
	ps_l2 = np.linalg.norm(np.divide(ps, np.sqrt(pr)))
	pt_l2 = np.linalg.norm(np.divide(pt, np.sqrt(pr)))
	return num / (ps_l2 * pt_l2)

def _feature(g, p0, tau, alpha):
	"""
	Given graph g, initial distribution p0, jump probability
	alpha, and walk cap tau. Returns random walk feature vector
	(size dependent only on tau).
	"""
	n, m = g.order(), []
	deg_inv = _degrees_inv(g)
	ps = _walks(g, _trans(g, deg_inv), p0, tau, alpha)
	pr = sp.csc_matrix(list(nx.pagerank(g, alpha = alpha).values())).T
	for s in range(tau):
		for t in range(s+1, tau+1):
			m += [_dist(pr, ps[s], ps[t])]
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
# POOLING FUNCTIONS
#

def pool_avg(alpha):
	"""
	Given list of vectors alpha. Returns entry-wise
	average.
	"""
	return np.average(alpha, axis=1)

def pool_max(alpha):
	"""
	Given list of vectors alpha. Returns entry-wise
	max.
	"""
	return np.amax(alpha, axis=1)

#
# WALK2VEC SPARSE CODING
#

def _all_features(g, tau, alpha):
	"""
	Given graph g, initial distribution generating function
	p0, and walk cap tau. Returns list of feature vectors
	over all vertices.
	"""
	n, xs = g.order(), []
	for i in range(n):
		xs += [_feature(g, _p0_dirac(g, i), tau, alpha)]
	return xs

def train(g, tau, alpha, d = None, params = {}):
	"""
	Given graph g and walk cap tau. Given optional dictionary
	d and training parameters params. Returns dictionary after
	training.
	"""
	if g.order() == 0:
		return d

	xs_mat = np.column_stack(_all_features(g, tau, alpha))
	xs = np.asfortranarray(xs_mat, dtype=float)
	default_params = {
		"K": 100,
		"lambda1": 0.15,
		"iter": 100,
		"batchsize": 5,
		"verbose": False
	}
	params = {**default_params, **params}
	return spams.trainDL(xs, **params)

def embed(g, tau, alpha, d, pool, params = {}):
	"""
	Given graph g, initial distribution generating function
	p0, and walk cap tau, dictionary d, and pooling function
	pool. Given optional coding parameters params. Returns
	embedding of g.
	"""
	atoms = d.shape[1]
	if g.order() == 0:
		return [0]*atoms

	xs_mat = np.column_stack(_all_features(g, tau, alpha))
	xs = np.asfortranarray(xs_mat, dtype=float)

	default_params = {
		"lambda1": 0.15
	}
	params = {**default_params, **params}

	a = spams.lasso(xs, D = d, **params).todense()
	return np.asarray(pool(a)).reshape(-1)
