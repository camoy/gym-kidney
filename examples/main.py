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
W_FUN = lambda o1, o2 : 1-0.5*(o1+o2)
#ACTION = actions.BloodAction(CYCLE_CAP, CHAIN_CAP, -4, 4, W_FUN)

# EMBEDDING CONSTS
#EMBEDDING = embeddings.OrderEmbedding()
#EMBEDDING = embeddings.NormalizeEmbedding(embeddings.OrderEmbedding(), [0.01])
#EMBEDDING = embeddings.UnionEmbedding([
#	embeddings.OrderEmbedding(),
#	embeddings.UnionEmbedding([
#	embeddings.OrderEmbedding(),
#	embeddings.OrderEmbedding()])
#	
#])
#EMBEDDING = embeddings.Walk2VecEmbedding([embeddings.p0_max, embeddings.p0_mean], 5, 0.05)
#EMBEDDING = embeddings.UnionEmbedding([embeddings.CycleFixedEmbedding(100, 2)])
#EMBEDDING = embeddings.UnionEmbedding([
#embeddings.CycleVariableEmbedding(3, 1000, 2)])
#EMBEDDING = embeddings.OrderEmbedding()

# MODEL CONSTS
M = 128
#K = 1024
K = 580
P = 0.05
P_A = 0.05
LEN = 3*K
#MODEL = models.HomogeneousModel(M, K, P, P_A, LEN)
MODEL = models.SparseModel(M, K, 0, "/home/camoy/tmp/unos_data_adj.csv", "/home/camoy/tmp/unos_data_details.csv", LEN)

#EMBEDDINGS
#EMBEDDING = embeddings.CycleVariableEmbedding(1, 3*M, 2)
#EMBEDDING = embeddings.CycleFixedEmbedding(M, 2)
#EMBEDDING = embeddings.UnionEmbedding(list(map(lambda i: embeddings.CycleFixedEmbedding(M, i), range(2, CYCLE_CAP+1))))

EMBEDDING = embeddings.UnionEmbedding([
embeddings.CycleFixedEmbedding(3*M, 2),
embeddings.ChainEmbedding(CHAIN_CAP),
embeddings.DdEmbedding(),
embeddings.NddEmbedding()])


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
			#print(env.action_space.sample())
			#obs, reward, done, _ = env.step(env.action_space.sample())#env.step(1)
			obs, reward, done, _ = env.step(0)
			print(obs)

if __name__ == "__main__":
	main()
