import gym
import gym_kidney

EPISODES = 250
FREQ = 10
PATH = "/home/user/greedy_kidney"

def main():
	env = gym.make("kidney-v0")
	env = gym_kidney.ConfigWrapper(env, "homogeneous", {
	        "rate": 25,
	        "k": 50,
	        "p": 0.05,
	        "p_a": 0.01
	})
	env = gym_kidney.LogWrapper(env, PATH, FREQ) 

	for i in range(EPISODES):
		obs, done = env.reset(), False
		while not done:
			_, _, done, _ = env.step(1)

if __name__ == "__main__":
	main()
