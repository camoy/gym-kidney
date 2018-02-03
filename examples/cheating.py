import gym
import gym_kidney
from gym_kidney import actions, embeddings, \
	models, loggers, wrappers

# LOCAL CONSTS
EPISODES = 100
SHOW = True
SEED = 3623451898

# ACTION CONSTS
CYCLE_CAP = 3
CHAIN_CAP = 3
ACTION = actions.FlapAction(CYCLE_CAP, CHAIN_CAP)

# EMBEDDING CONSTS
EMBEDDING = embeddings.OmniscientEmbedding()

# MODEL CONSTS
M = 40
K = 24
#K = 580
P = 0.05
P_A = 0.05
LEN = 30
#MODEL = models.HomogeneousModel(M, K, P, P_A, LEN)
MODEL = models.OmniscientModel(M, K, 0.7, "/home/camoy/tmp/unos_data_adj.csv", "/home/camoy/tmp/unos_data_details.csv", LEN)

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
			act = 1 if obs[0] > 0.0 else 0
			print("ACT:", act)
			obs, reward, done, _ = env.step(act)
			print(obs)

if __name__ == "__main__":
	main()
