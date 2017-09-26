# Loggers

Loggers allow one to record the performance of agents operating under
different environmental conditions. This works by pooling together
the `params` and `stats` dictionaries from respective components of
the environment.

## `CsvEmbedding`

Outputs experimental results as two CSV files. One records the parameters
of the environment, the other the performance of the agent after interacting
with the environment.
