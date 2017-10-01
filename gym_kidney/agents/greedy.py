from gym_kidney import agents
from gym import spaces

#
# GreedyAgent attempts a maximum cardinality matching at every
# time step. This requires a discrete action space.
#
class GreedyAgent(agents.Agent):

	def act(self, env, obs, done):
		action_space = env.action_space

		if isinstance(action_space, spaces.Discrete):
			return 1
		else:
			raise TypeError("cannot be greedy on continuous space")
