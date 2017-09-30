#
# Logger is an abstract class to be implemented by all loggers
# recording the environment.
#
class Logger:
	# output_log : Env -> None
	# Outputs logged information to a file
	def output_log(self, env):
		raise NotImplementedError

	# _flush_stat : Env -> None
	# Clears all statistics
	def _flush_stat(self, env):
		components = [env.action, env.agent, env.embedding, env.model]

		for component in components:
			for k, _ in component.stats.items():
				component.stats[k] = 0
