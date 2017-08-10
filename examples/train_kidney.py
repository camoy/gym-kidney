import gym
import gym_kidney
from baselines import deepq

EPISODES = 1000
NN = "dqn"
EXP = 0
OUT = "/home/user/"
FREQ = 10
PARAM = {
	"layers": [64, 64, 64],
	"learning_rate": 1e-3,
	"buffer_size": 50000,
	"exploration_fraction": 0.1,
	"exploration_final_eps": 0.02
}

def callback(lcl, glb):
	return len(lcl["episode_rewards"]) >= EPISODES

def main():
	env = gym.make("kidney-v0")
	env = gym_kidney.ConfigWrapper(env, "kidney", {
	        "m": 580,
	        "k": 24,
		"t": 3,
		"data": "/home/user/data_adj.csv",
		"details": "/home/user/data_details.csv",
		"embed": {
			"method": "walk2vec-sc",
			"tau": 5,
			"d_path": "/home/user/dictionary.gz"
		}
	})
	env = gym_kidney.LogWrapper(env, NN, EXP, OUT, FREQ, PARAM)
	model = deepq.models.mlp(PARAM["layers"])
	act = deepq.learn(
		env,
		q_func = model,
		lr = PARAM["learning_rate"],
		max_timesteps = 1000000,
		buffer_size = PARAM["buffer_size"],
		exploration_fraction = PARAM["exploration_fraction"],
		exploration_final_eps = PARAM["exploration_final_eps"],
		print_freq = 10,
		callback = callback
	)
	print("Saving model to kidney_model.pkl")
	act.save("kidney_model.pkl")

if __name__ == '__main__':
	main()
