from gym_kidney import embeddings
from gym import spaces

import math
import networkx as nx
import numpy as np
import scipy.sparse as sp
# import spams

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

def _kl_sym_div(p, q):
	with np.errstate(divide = "ignore", invalid = "ignore"):
		pq_log = np.ma.log(np.nan_to_num(p / q))
		qp_log = np.ma.log(np.nan_to_num(q / p))
		s1 = p.T.dot(pq_log.filled(0)).item(0)
		s2 = q.T.dot(qp_log.filled(0)).item(0)
		return s1 + s2

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
			m += [_kl_sym_div(ps[s], ps[t])]
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

# A DistrFun is one of:
# - p0_min, Dirac delta distribution at vertex of min degree
# - p0_max, Dirac delta distribution at vertex of max egree
# - p0_median, Dirac delta distribution at vertex of median degree
# - p0_mean, Dirac delta distribution at vertex of mean degree

#
# Walk2VecEmbedding embeds the graph according to a modified Walk2Vec
# random walk method. It it parametrized by:
# - p0s : [DistrFun], initial distributions
# - tau : Nat, steps in the random walk
# - alpha : (0, 1], jump probability
#
class Walk2VecEmbedding(embeddings.Embedding):
	def __init__(self, p0s, tau, alpha):
		self.p0s = p0s
		self.tau = tau
		self.alpha = alpha
		self.size = int(len(p0s)*((tau**2+tau)/2))

		self.params = {
			"tau": tau,
			"alpha": alpha
		}

		self.observation_space = spaces.Box(0, np.inf, (self.size,))

	def embed(self, g, rng):
		"""
		Given graph g, list of initial distribution generating
		functions p0s, jump probability alpha,and walk cap tau.
		Returns embedding of g.
		"""
		p0s = self.p0s
		tau = self.tau
		alpha = self.alpha

		# empty graph
		if g.order() == 0:
			return np.array([0]*self.size)

		# non-empty graphs
		phi = []
		for i, p0_i in enumerate(p0s):
			phi += _feature(g, p0_i(g), tau, alpha)
		return np.array(phi)

#
# WALK2VEC SPARSE CODING
#

'''

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

class Walk2VecScEmbedding(embeddings.Embedding):

	def __init__(self, tau, alpha, d, pool, param_coding):
		self.tau = tau
		self.alpha = alpha
		self.d = d 
		self.pool = pool 
		self.param_coding = param_coding

		self.params = {
			"tau": tau,
			"alpha": alpha
		}

	def embed(self, g, rng):
		"""
		Given graph g, initial distribution generating function
		p0, and walk cap tau, dictionary d, and pooling function
		pool. Given optional coding parameters params. Returns
		embedding of g.
		"""
		tau = self.tau
		alpha = self.alpha
		d = self.d 
		pool = self.pool 
		param_coding = self.param_coding
		atoms = d.shape[1]

		if g.order() == 0:
			return np.array([0]*atoms)

		xs_mat = np.column_stack(_all_features(g, tau, alpha))
		xs = np.asfortranarray(xs_mat, dtype=float)

		default_params = {
			"lambda1": 0.15
		}
		param_coding = {**default_params, **param_coding}

		a = spams.lasso(xs, D = d, **param_coding).todense()
		return np.asarray(pool(a)).reshape(-1)

	def train(self, g):
		"""
		Given graph g and walk cap tau. Given optional dictionary
		d and training parameters params. Returns dictionary after
		training.
		"""
		tau = self.tau
		alpha = self.alpha
		d = self.d 
		pool = self.pool 
		param_coding = self.param_coding

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
		param_coding = {**default_params, **param_coding}
		return spams.trainDL(xs, **param_coding)
'''
