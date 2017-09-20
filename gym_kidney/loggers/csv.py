from gym_kidney import loggers

class CsvLogger(loggers.Logger):

	def __init__(self, path, exp):
		self.path = path
		self.exp = exp
		self.param_file = "%s/%03d_%s.csv" % (path, exp, "PARAM")
		self.stat_file = "%s/%03d_%s.csv" % (path, exp, "STAT")
		self.output_headers = False

	def output_log(self, env):
		components = [env.action, env.embedding, env.model]
		params = {}
		stats = {}

		for component in components:
			params = {**params, **component.params}
			stats = {**stats, **component.stats}

		if not self.output_headers:
			with open(self.param_file, "w") as f:
				self._write_array(f, params.keys())
				self._write_array(f, params.values())

			with open(self.stat_file, "w") as f:
				self._write_array(f, stats.keys())

			self.output_headers = True

		with open(self.stat_file, "a") as f:
			self._write_array(f, stats.values())

		self._flush_stat(env)

	def _write_array(self, f, array):
		array = map(str, array)
		f.write("%s\n" % (",".join(array)))
