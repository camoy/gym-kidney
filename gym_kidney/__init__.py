from gym.envs.registration import register

register(
	id = "kidney-v0",
	entry_point = "gym_kidney.envs:KidneyEnv",
	max_episode_steps = 100000
)
