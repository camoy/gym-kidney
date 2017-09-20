class CsvLogger(Logger):

	def __init__(path, exp):
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
			params = {**params, component.params}
			stats = {**stats, component.stats}

		if not self.output_headers:
			with open(self.param_file, "w") as f:
				_write_array(f, params.keys)
				_write_array(f, params.values)

			with open(self.stat_file, "w") as f:
				_write_array(f, stats.keys)

			self.output_headers = True

		with open(self.stat_file, "a") as f:
			_write_array(f, stats.values)

		self._flush_stat()

	def _write_array(f, array):
		f.write("%s\n" % (",".join(array)))
