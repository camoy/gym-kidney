# Loggers

Loggers allow one to record the performance of agents operating under
different environmental conditions. This works by pooling together
the `params` and `stats` dictionaries from respective components of
the environment.

## `CsvLogger`

`CsvLogger` outputs results as two CSV files. One records the parameters
of the environment, the other the performance of the agent after interacting
with the environment.

* `path : String`, the directory the file will go
* `exp : Nat`, the number of the experiment
* `custom : Dict`, dictionary of custom parameters
