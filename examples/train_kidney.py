import gym
import gym_kidney
from baselines import deepq

def callback(lcl, glb):
	return len(lcl["episode_rewards"]) >= 1000

def main():
	env = gym.make("kidney-v0")
	env = gym_kidney.ConfigWrapper(env, "homogeneous", {
	        "m": 1460,
	        "k": 24,
	        "d": 326,
		"t": 5,
	        "p_a": 0.01
	})
	model = deepq.models.mlp([64,64,64])
	act = deepq.learn(
		env,
		q_func = model,
		lr = 1e-3,
		max_timesteps = 1000000,
		buffer_size = 50000,
		exploration_fraction = 0.1,
		exploration_final_eps = 0.02,
		print_freq = 10,
		callback = callback
	)
	print("Saving model to kidney_model.pkl")
	act.save("kidney_model.pkl")

if __name__ == '__main__':
	main()
