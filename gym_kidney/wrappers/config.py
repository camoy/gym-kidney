import gym
from gym import Wrapper
from gym_kidney.envs import kidney_common as kc

init_distrs = {
	"max": kc.p0_max,
	"min": kc.p0_min,
	"mean": kc.p0_mean,
	"median": kc.p0_median
}

models = {
	"contrived": kc.ContrivedModel,
	"homogeneous": kc.HomogeneousModel,
	"heterogeneous": kc.HeterogeneousModel,
	"kidney": kc.KidneyModel
}

class ConfigWrapper(Wrapper):
	def __init__(self, env, model, p):
		super(ConfigWrapper, self).__init__(env)

		# environment parameters
		if "tau" in p: self.env.tau = p.pop("tau")
		if "alpha" in p: self.env.alpha = p.pop("alpha")
		if "eps_len" in p: self.env.eps_len = p.pop("eps_len")
		if "cycle_cap" in p: self.env.cycle_cap = p.pop("cycle_cap")
		if "chain_cap" in p: self.env.chain_cap = p.pop("chain_cap")
		if "init_distrs" in p:
			getter, distrs = init_distrs.get, p.pop("init_distrs")
			self.env.init_distrs = list(map(getter, distrs))

		# model parameters
		p["rng"] = self.env.rng
		self.env.model = models[model](**p)
