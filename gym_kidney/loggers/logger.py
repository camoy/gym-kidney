class Logger:

	# params :: Dictionary
	params = {}

	# stats :: Dictionary
	stats = {}

	# output_log :: Environment -> None
	def output_log(self, env):
		raise NotImplementedError

	# _flush_stat :: Environment -> None
	def _flush_stat(self, env):
		components = [env.action, env.embedding, env.model]

		for component in components:
			for k, _ in component.stats.items():
				component.stats[k] = 0
