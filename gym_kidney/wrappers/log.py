import gym
from gym import Wrapper

class LogWrapper(Wrapper):
	def __init__(self, env, nn, exp, path, freq, param_dict):
		super(LogWrapper, self).__init__(env)

		env = env.unwrapped
		self._exp = exp
		self._freq = freq
		self._net_reward = 0
		self._eps = 0
		self._iter = 0
		self._data_path = "%s/%s_%s_data.csv" % (path, nn, exp)
		self._param_path = "%s/%s_%s_param.csv" % (path, nn, exp)

		# data header
		keys = [
			"iteration",
			"net_reward"
		] + list(env.model.logd.keys())
		with open(self._data_path, "w") as f:
			f.write("%s\n" % (",".join(keys)))

		# param
		keys = [
			"seed",
			"tau",
			"cycle_cap",
			"chain_cap"
			"episodes"
		] + list(param_dict.keys())

		values = list(map(str, [
			env.seed,
			env.tau,
			env.cycle_cap,
			env.chain_cap,
			self._eps
		] + list(param_dict.values())))

		with open(self._param_path, "w") as f:
			f.write("%s\n" % (",".join(keys)))
			f.write("%s\n" % (",".join(values)))

	def _log_csv(self):
		"""
		Appends average reward per episode to log file.
		"""
		env = self.env.unwrapped
		params = [
			self._iter,
			self._net_reward,
			self._eps
		]
		params = params + env.model.log + list(env.model.logd.values())
		params = list(map(str, params))
		with open(self._data_path, "a") as f:
			f.write("%s\n" % (",".join(params)))

	def _step(self, action):
		"""
		Wraps step.
		"""
		obs, reward, done, info = self.env.step(action)
		self._net_reward += reward
		return obs, reward, done, info

	def _reset(self):
		"""
		Wraps reset.
		"""
		self._eps += 1
		if self._eps >= self._freq:
			self._log_csv()
			self._eps = 0
			self._net_reward = 0
			self._iter += 1
		return self.env.reset()
