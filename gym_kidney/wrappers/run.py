import gym
from gym import Wrapper

#
# RunWrapper runs the agent on the environment.
#
class RunWrapper(Wrapper):

	def __init__(self, env, eps, show):
		env = env.unwrapped
		self.eps = eps
		self.show = show
		super(RunWrapper, self).__init__(env)

	# run : Env, Nat -> None
	# Runs the agent in the environment
	def run(self):
		for i in range(self.eps):
			self._run_episode()

	# _run_episode : None -> None
	# Runs the agent in the environment for one episode
	def _run_episode(self):
		env = self.env
		obs, done = env.reset(), False
		while not done:
			action = env.agent.act(env, obs, done)
			obs, reward, done, _ = env.step(action)
			if self.show:
				env.render()
