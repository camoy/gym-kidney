import gym
from gym import Wrapper

#
# ConfigWrapper initializes the environment with the given components.
#
class ConfigWrapper(Wrapper):

	def __init__(self, env, action, embedding, model, logger):
		env = env.unwrapped
		env.action = action
		env.embedding = embedding
		env.model = model
		env.logger = logger
		action.env = embedding.env = model.env = logger.env = env
		env.setup()

		super(ConfigWrapper, self).__init__(env)
