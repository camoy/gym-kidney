import gym
import gym_kidney

EPISODES = 1000
NN = "random"
EXP = 0
OUT = "/home/user/"
FREQ = 10
PARAM = {}

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

	for i in range(EPISODES):
		obs, done = env.reset(), False
		while not done:
			_, _, done, _ = env.step(env.action_space.sample())

if __name__ == "__main__":
	main()
