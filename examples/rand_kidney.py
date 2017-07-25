import gym
import gym_kidney

EPISODES = 250
FREQ = 10
OUT = "/home/user/rand_kidney"

def main():
	env = gym.make("kidney-v0")
	env = gym_kidney.ConfigWrapper(env, "homogeneous", {
	        "m": 1460,
	        "k": 24,
	        "d": 326,
		"t": 5,
	        "p_a": 0.01
	})
	env = gym_kidney.LogWrapper(env, OUT, FREQ) 

	for i in range(EPISODES):
		obs, done = env.reset(), False
		while not done:
			_, _, done, _ = env.step(env.action_space.sample())

if __name__ == "__main__":
	main()
