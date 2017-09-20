import gym
from gym import Wrapper

class ConfigWrapper(Wrapper):

	def __init__(self, env, action, embedding, model, logging):
		env = env.unwrapped
		env.action = action
		env.embedding = embedding
		env.model = model
		env.logging = logging
		env.setup()

		super(ConfigWrapper, self).__init__(env)
