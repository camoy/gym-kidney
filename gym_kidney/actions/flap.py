from gym import spaces
from gym_kidney import actions
from gym_kidney import _solver

#
# FlapAction performs a maximum cardinality matching on
# the graph. It is parametrized by:
# - (cycle_cap :: Nat) the cycle cap for the solver
# - (chain_cap :: Nat) the chain cap for the solver
#
class FlapAction(actions.Action):

	action_space = spaces.Discrete(2)

	def __init__(self, cycle_cap, chain_cap):
		self.cycle_cap = cycle_cap
		self.chain_cap = chain_cap

		self.params = {
			"cycle_cap": cycle_cap,
			"chain_cap": chain_cap
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
		G = self._process_matches(G, M)

		rew_cycles = sum(map(lambda x: len(x), soln.cycles))
		rew_chains = sum(map(lambda x: len(x.vtx_indices), soln.chains))
		reward = rew_cycles + rew_chains

		self.stats["cycle_reward"] += rew_cycles
		self.stats["chain_reward"] += rew_chains

		return (G, reward)
