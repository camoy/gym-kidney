from gym.envs.registration import register
from gym_kidney.wrappers.config import ConfigWrapper

register(
	id = "kidney-v0",
	entry_point = "gym_kidney.envs:KidneyEnv",
)
