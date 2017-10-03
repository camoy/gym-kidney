from gym import spaces
from gym_kidney import actions
from gym_kidney import _solver

BLOODS = {
	"A": 0,
	"B": 1,
	"AB": 2,
	"O": 3,
	"-": 4
}

#
# BloodAction reweights the graph edges according to the
# the action before calling the solver. It is parametrized
# by:
# - cycle_cap : Nat, the cycle cap for the solver
# - chain_cap : Nat, the chain cap for the solver
# - min : Real, smallest value for vertex
# - max : Real, largest value for vertex
# - w_fun : (Real, Real -> Real), weight function
#
class BloodAction(actions.Action):

	def __init__(self, cycle_cap, chain_cap, min, max, w_fun):
		self.cycle_cap = cycle_cap
		self.chain_cap = chain_cap
		self.min = min
		self.max = max
		self.w_fun = w_fun
		self.action_space = spaces.Box(min, max, (len(BLOODS)**2,))

		self.params = {
			"cycle_cap": cycle_cap,
			"chain_cap": chain_cap,
			"min": min,
			"max": max
		}

		self.stats = {
			"cycle_reward": 0,
			"chain_reward": 0
		}


	def do_action(self, G, action):
		if action == 0:
			return (G, 0)

		dd, ndd = self._nx_to_ks(G)
		cfg = _solver.kidney_ip.OptConfig(
			dd,
			ndd,
			self.cycle_cap,
			self.chain_cap)
		soln = _solver.solve_kep(cfg, "picef")
		M = (soln.cycles, soln.chains)
		G = self._reweight(G, action)
		G = self._process_matches(G, M)

		rew_cycles = sum(map(lambda x: len(x), soln.cycles))
		rew_chains = sum(map(lambda x: len(x.vtx_indices), soln.chains))
		reward = rew_cycles + rew_chains

		self.stats["cycle_reward"] += rew_cycles
		self.stats["chain_reward"] += rew_chains

		return (G, reward)

	def _reweight(self, G, action):
		for u, v, d in G.edges(data = True):
			o1 = self._vertex_weight(G, u, action)
			o2 = self._vertex_weight(G, v, action)
			d["weight"] = self.w_fun(o1, o2)
			#d["weight"] = 1 - 0.5*(o1+o2)
		return G 

	def _vertex_weight(self, G, u, action):
		bd, bp = G.node[u]["bd"], G.node[u]["bp"]
		return action[BLOODS[bd]*len(BLOODS) + BLOODS[bp]]
