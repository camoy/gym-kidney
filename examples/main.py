import gym
import gym_kidney
from gym_kidney import actions, agents, embeddings, \
	models, loggers, wrappers

# LOCAL CONSTS
EPISODES = 100

# AGENT CONSTS
AGENT = agents.GreedyAgent()

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
	AGENT.run(env, EPISODES)

if __name__ == "__main__":
	main()
