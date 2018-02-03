import gym
import gym_kidney
from gym_kidney import actions, embeddings, \
	models, loggers, wrappers

# LOCAL CONSTS
EPISODES = 10

# EMBEDDING CONSTS
#EMBEDDING = embeddings.NopEmbedding()
EMBEDDING = embeddings.NormalizeEmbedding(embeddings.UnionEmbedding([embeddings.CycleFixedEmbedding(10,10),embeddings.CycleVariableEmbedding(1,10,10),]), [1,1])

# MODEL CONSTS
M = 64

# LOGGING CONSTS
PATH = "/home/camoy/tmp/"
CUSTOM = { "agent" : "greedy" }

# MAIN
def main():
	# PARAMETERS
	KS = [32, 128, 512]
	CYCLES = [3, 4, 5]
	CHAINS = [3, 6, 9]
	PARAMS = zip(KS, CYCLES, CHAINS)
	EXP = 0

	for K, CYCLE_CAP, CHAIN_CAP in PARAMS:
		LOGGING = loggers.CsvLogger(PATH, EXP, CUSTOM)
		LEN = 3*K
		MODEL = models.DataModel(M, K, "/home/camoy/tmp/unos_data_adj.csv", "/home/camoy/tmp/unos_data_details.csv", LEN)
		ACTION = actions.FlapAction(CYCLE_CAP, CHAIN_CAP)

		env = gym.make("kidney-v0")
		env = wrappers.ConfigWrapper(env, ACTION, EMBEDDING, MODEL, LOGGING)
		for i in range(EPISODES):
			obs, done = env.reset(), False
			while not done:
				env.render()
				obs, reward, done, _ = env.step(1)
		EXP += 1

if __name__ == "__main__":
	main()
