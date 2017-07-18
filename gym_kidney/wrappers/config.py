import gym
from gym import Wrapper
from gym_kidney.envs import kidney_common as kc

class ConfigWrapper(Wrapper):
	def __init__(self, env, model, params):
		super(ConfigWrapper, self).__init__(env)

		params["rng"] = self.env.rng
		if model == "homogeneous":
			self.env.model = kc.HomogeneousModel(**params)
