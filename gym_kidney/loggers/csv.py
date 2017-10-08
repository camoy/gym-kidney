from gym_kidney import loggers

# CsvLogger outputs results as two CSV files. One records the parameters
# of the environment, the other the performance of the agent after interacting
# with the environment.
# - path : String, the directory the file will go
# - exp : Nat, the number of the experiment
# - custom : Dict, dictionary of custom parameters
#
class CsvLogger(loggers.Logger):

	def __init__(self, path, exp, custom = {}):
		self.path = path
		self.exp = exp
		self.custom = custom
		self.param_file = "%s/%03d_%s.csv" % (path, exp, "PARAM")
		self.stat_file = "%s/%03d_%s.csv" % (path, exp, "STAT")
		self.output_headers = False

	def output_log(self, env):
		components = [env.action, env.embedding, env.model]
		params = {
			"action" : type(env.action).__name__,
			"embedding" : type(env.embedding).__name__,
			"model" : type(env.model).__name__,
			"seed" : env.rng_seed
		}
		stats = {}

		for component in components:
			params = {**params, **component.params, **self.custom}
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
