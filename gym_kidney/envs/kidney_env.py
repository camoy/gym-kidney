import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym_kidney.envs import kidney_solver as ks
from gym_kidney.envs import kidney_common as kc

import numpy as np
import networkx as nx
import os

class KidneyEnv(gym.Env):
	metadata = { "render.modes" : ["human"] }

	def __init__(self):
		# default parameters
		self.cycle_cap = 3
		self.chain_cap = 3
		self.t = 5
		self.embed = {
			#"atoms": 100,
			"method": "walk2vec",
			"tau": 7,
			"alpha": 0.05,
			"init_distrs": [kc.p0_max, kc.p0_mean]#,
			#"d_path": "/home/camoy/tmp/dictionary.gz",
			#"training": False
		}
		self.dict = None
		self.lembed = []

		# initialize
		self._seed()
		self.model = kc.ContrivedModel(self.rng)
		self._setup()
		self._reset()

	def _setup(self):
		m = self.embed["method"]

		if m == "walk2vec":
			tau = self.embed["tau"]
			inits = self.embed["init_distrs"]
			obs_size = int(((tau**2+tau)/2)*len(inits))
		else:
			obs_size = self.embed["atoms"]
			path = self.embed["d_path"]

			# dictionary
			if path and os.path.exists(path):
				self.dict = np.loadtxt(path)
				self.dict = np.asfortranarray(self.dict, dtype = float)
			else:
				self.dict = None

		# spaces
		self.action_space = spaces.Box(
			-4,
			4,
			(25,))
		self.observation_space = spaces.Box(
			-np.inf,
			np.inf,
			(obs_size,))

		# embedding
		self.lembed = [0] * obs_size

		# length
		self.eps_len = self.model.k * self.t

		# seed
		self._seed(seed = self.seed)

	def _seed(self, seed = None):
		self.seed = seed
		self.rng, seed = seeding.np_random(seed)
		return [seed]

	def _step(self, action):
		reward = 0
		match = ([], [])

		action = list(map(lambda x: 0 if np.isnan(x) else x, action))
		action = np.array(action)

		self.graph = kc.reweight(self.graph, action)
		d, ndds = kc.nx_to_ks(self.graph)
		cfg = ks.kidney_ip.OptConfig(
			d,
			ndds,
			self.cycle_cap,
			self.chain_cap)
		soln = ks.solve_kep(cfg, "picef")
		match = (soln.cycles, soln.chains)

		# utility as cardinality
		rew_cycles = sum(map(lambda x: len(x), soln.cycles))
		rew_chains = sum(map(lambda x: len(x.vtx_indices), soln.chains))
		reward = rew_cycles + rew_chains

		# evolve
		self.changed, self.graph = self.model.evolve(
			self.graph,
			match,
			self.tick)
		self.tick = self.tick + 1

		# return values
		embedding = self._get_obs()
		done = self.tick >= self.eps_len
		return embedding, reward, done, {}

	def _reset(self):
		# state
		self.tick = 0
		self.changed = True
		self.graph = self.model.reset()

		# dictionary
		if not (self.dict is None) and self.embed["training"]:
			np.savetxt(self.d_path, self.dict)

		return self._get_obs()

	def _get_obs(self):
		graph = self.graph
		tau = self.embed["tau"]
		alpha = self.embed["alpha"]
		inits = self.embed["init_distrs"]
		m = self.embed["method"]

		# new embedding
		if not (m == "walk2vec") and self.embed["training"]:
			atoms = self.embed["atoms"]
			self.dict = kc.train(
				graph,
				tau,
				alpha,
				d = self.dict,
				params = { "K": atoms })
		elif self.changed:
			self.changed = False
			if m == "walk2vec":
				self.lembed = kc.phi(
					graph,
					inits,
					tau,
					alpha)
			else:
				self.lembed = kc.phi_sc(
					graph,
					tau,
					alpha,
					self.dict,
					kc.pool_avg)

		# return cached embedding
		return np.array(self.lembed)

	def _render(self, mode = "human", close = False):
		if close:
			return

		import matplotlib.pyplot as plt

		if self.tick == 0:
			plt.ion()

		# define colors
		g = self.graph
		attrs = nx.get_node_attributes(g, "ndd")
		values = ["red" if attrs[v] else "blue" for v in g.nodes()]

		# draw graph
		plt.clf()
		nx.draw(g,
			pos = nx.circular_layout(g),
			node_color = values)
		plt.pause(0.01)

		return []
