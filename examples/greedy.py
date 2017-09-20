import gym
import gym_kidney

from gym_kidney import actions
from gym_kidney import embeddings
from gym_kidney import models
from gym_kidney import loggers
from gym_kidney import wrappers

# LOCAL CONSTS
EPISODES = 100

# ACTION CONSTS
CYCLE_CAP = 3
CHAIN_CAP = 3
ACTION = actions.FlapAction(CYCLE_CAP, CHAIN_CAP)

# EMBEDDING CONSTS
EMBEDDING = embeddings.OrderEmbedding()

# MODEL CONSTS
M = 580
K = 24 
P = 0.05
P_A = 0.05
LEN = 200
MODEL = models.HomogeneousModel(M, K, P, P_A, LEN)

# LOGGING CONSTS
PATH = "/home/camoy/tmp/"
EXP = 0
LOGGING = loggers.CsvLogger(PATH, EXP)

# MAIN
def main():
	env = gym.make("kidney-v0")
	env = wrappers.ConfigWrapper(env, ACTION, EMBEDDING, MODEL, LOGGING)

	for i in range(EPISODES):
		obs, done = env.reset(), False
		while not done:
			_, _, done, _ = env.step([1])
			env.render()

if __name__ == "__main__":
	main()
