from gym.envs.registration import register
#from gym_kidney.wrappers.config import ConfigWrapper
#from gym_kidney.wrappers.log import LogWrapper

register(
	id = "kidney-v0",
	entry_point = "gym_kidney.envs:KidneyEnv",
)
