#
# Agent is an abstract class to be implemented by all agents
# interacting with the environment.
#
class Agent:

	# params : Dict
	params = {}

	# stats : Dict
	stats = {}

	# act : Env, NP Array -> Action
	# Decides on an action based on an observation
	def act(self, env, obs):
		raise NotImplementedError

