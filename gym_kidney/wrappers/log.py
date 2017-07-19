import gym
from gym import Wrapper

class LogWrapper(Wrapper):
	def __init__(self, env, path, freq):
		super(LogWrapper, self).__init__(env)

		self._path = path
		self._freq = freq
		self._net_reward = 0
		self._eps = 0
		self._iter = 0

	def _log_csv(self):
		avg_reward = self._net_reward / self._eps
		env = self.env.unwrapped
		params = [self._iter, env.seed, avg_reward] + env.model.log
		params = list(map(str, params))
		with open(self._path, "a") as f:
			f.write("%s\n" % (",".join(params)))

	def _step(self, action):
		obs, reward, done, info = self.env.step(action)
		self._net_reward += reward
		return obs, reward, done, info

	def _reset(self):
		self._eps += 1
		if self._eps >= self._freq:
			self._log_csv()
			self._eps = 0
			self._net_reward = 0
			self._iter += 1
		return self.env.reset()
