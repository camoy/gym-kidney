#
# Agent is an abstract class to be implemented by all agents
# interacting with the environment.
#
class Agent:

	# run :: Env, Nat -> None
	# Runs the agent in the environment
	def run(self, env, eps):
		for i in range(eps):
			self._run_episode(env)

	# _run_episode :: Env -> None
	# Runs the agent in the environment for one episode
	def _run_episode(self, env):
		obs, done = env.reset(), False
		while not done:
			action = self.act(env, obs)
			obs, reward, done, _ = env.step(action)

	# act :: Env, NumPy Array -> Action
	# Decides on an action based on an observation
	def act(self, env, obs):
		raise NotImplementedError

