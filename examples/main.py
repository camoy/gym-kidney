import gym
import gym_kidney
from gym_kidney import actions, embeddings, \
	models, loggers, wrappers

# LOCAL CONSTS
EPISODES = 100
SHOW = False

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
CUSTOM = { "agent" : "greedy" }
LOGGING = loggers.CsvLogger(PATH, EXP, CUSTOM)

# MAIN
def main():
	env = gym.make("kidney-v0")
	env = wrappers.ConfigWrapper(env, ACTION, EMBEDDING, MODEL, LOGGING)
	for i in range(EPISODES):
		obs, done = env.reset(), False
		while not done:
			if SHOW:
				env.render()
			obs, reward, done, _ = env.step(1)

if __name__ == "__main__":
	main()
