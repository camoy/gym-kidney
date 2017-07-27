import gym
import gym_kidney

EPISODES = 1000
NN = "greedy"
EXP = 0
OUT = "/home/user/"
FREQ = 10
PARAM = {}

def main():
	env = gym.make("kidney-v0")
	env = gym_kidney.ConfigWrapper(env, "kidney", {
		"tau": 5,
	        "m": 580,
	        "k": 24,
		"t": 3,
		"data": "/home/user/data_adj.csv",
		"details": "/home/user/data_details.csv"
	})
	env = gym_kidney.LogWrapper(env, NN, EXP, OUT, FREQ, PARAM)

	for i in range(EPISODES):
		obs, done = env.reset(), False
		while not done:
			_, _, done, _ = env.step(1)

if __name__ == "__main__":
	main()
