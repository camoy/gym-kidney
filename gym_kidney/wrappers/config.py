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
		env = env.unwrapped

		# environment parameters
		if "seed" in p: env.seed = p.pop("seed")
		if "tau" in p: env.tau = p.pop("tau")
		if "alpha" in p: env.alpha = p.pop("alpha")
		if "t" in p: env.t = p.pop("t")
		if "cycle_cap" in p: env.cycle_cap = p.pop("cycle_cap")
		if "chain_cap" in p: env.chain_cap = p.pop("chain_cap")
		if "init_distrs" in p:
			getter, distrs = init_distrs.get, p.pop("init_distrs")
			env.init_distrs = list(map(getter, distrs))

		# model parameters
		p["rng"] = env.rng
		env.model = models[model](**p)

		# initialize
		env._setup()
		super(ConfigWrapper, self).__init__(env)
