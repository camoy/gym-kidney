import gym
from gym import Wrapper

#
# ConfigWrapper initializes the environment with the given components.
#
class ConfigWrapper(Wrapper):

	def __init__(self, env, action, agent, embedding, model, logger):
		env = env.unwrapped
		env.action = action
		env.agent = agent
		env.embedding = embedding
		env.model = model
		env.logger = logger
		env.setup()

		super(ConfigWrapper, self).__init__(env)
