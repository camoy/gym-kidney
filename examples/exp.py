import gym
import gym_kidney
from gym_kidney import actions, embeddings, \
	models, loggers, wrappers

# LOCAL CONSTS
EPISODES = 20
SHOW = False

# ACTION CONSTS
CYCLE_CAP = 2
CHAIN_CAP = 1
ACTION = actions.FlapAction(CYCLE_CAP, CHAIN_CAP)

# EMBEDDING CONSTS
#EMBEDDING = embeddings.OrderEmbedding()
EMBEDDING = embeddings.NopEmbedding()

# MODEL CONSTS
N = 1000
RHO = 1
P_H = 0.05
P_L = 0
MODEL = models.AshlagiModel(N, RHO, P_H, P_L)

# LOGGING CONSTS
PATH = "/home/camoy/tmp/ashlagi"
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
