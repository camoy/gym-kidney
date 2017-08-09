import gym
import gym_kidney

EPISODES = 1000
NN = "contrived"
EXP = 0
OUT = "/home/user/"
FREQ = 10
PARAM = {}

def main():
	env = gym.make("kidney-v0")
	env = gym_kidney.ConfigWrapper(env, "contrived", {})

	for i in range(EPISODES):
		obs, done = env.reset(), False
		while not done:
			_, rew, done, _ = env.step([2])

if __name__ == "__main__":
	main()
