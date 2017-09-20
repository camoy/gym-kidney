import gym
import gym_kidney

# ACTION CONSTS
CYCLE_CAP = 3
CHAIN_CAP = 3
ACTION = gym_kidney.actions.FlapAction(CYCLE_CAP, CHAIN_CAP)

# EMBEDDING CONSTS
EMBEDDING = gym_kidney.embeddings.OrderEmbedding()

# MODEL CONSTS
M = 580
K = 24 
P = 0.05
P_A = 0.05
LEN = 200
MODEL = gym_kidney.models.HomogeneousModel(M, K, P, P_A, LEN)

# LOGGING CONSTS
PATH = "/home/camoy/tmp/"
EXP = 0
LOGGING = gym_kidney.loggers.CsvLogger(PATH, EXP)

# MAIN
def main():
	env = gym.make("kidney-v0")
	env = gym_kidney.ConfigWrapper(env, ACTION, EMBEDDING, MODEL, LOGGING)

	for i in range(EPISODES):
		obs, done = env.reset(), False
		while not done:
			_, _, done, _ = env.step(1)

if __name__ == "__main__":
	main()
