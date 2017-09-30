from gym_kidney import agents
from gym import spaces

#
# RandomAgent chooses a random action from the
# action space.
#
class RandomAgent(agents.Agent):

	def act(self, env, obs):
		action_space = env.action_space
		return action_space.sample()
