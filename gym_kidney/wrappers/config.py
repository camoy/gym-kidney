import gym
from gym import Wrapper

class ConfigWrapper(Wrapper):

	def __init__(self, env, action, embedding, model, logger):
		env = env.unwrapped
		env.action = action
		env.embedding = embedding
		env.model = model
		env.logger = logger
		env.setup()

		super(ConfigWrapper, self).__init__(env)
