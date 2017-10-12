# Embeddings

Embeddings define how the agent observes the environment. It must convert
the graph into a fixed-sized vector representation. There are two types
of embeddings possible: atomic and composite.

Atomic embeddings operate on the graph itself. Composite embeddings operate
on other embeddings. Currently, the only composite embedding is
`UnionEmbedding`.

## `ChainEmbedding`

`ChainEmbedding` embeds the sum of longest chains possible from
all non-directed donors.

* `chain_length : Nat`, chain length under consideration

## `CycleFixedEmbedding`

`CycleFixedEmbedding` embeds an estimate for the number of cycles in the graph
using a fixed number of samples.

* `sample_size : Nat`, number of samples to take
* `cycle_length : Nat`, cycle length under consideration

## `CycleVariableEmbedding`

CycleVariableEmbedding embeds an estimate for the number of cycles in the
graph using a variable number of samples.

* `successes : Nat`, number of successes before stopping
* `sample_cap : Nat`, maximum samples to take before quitting
* `cycle_length : Nat`, cycle length under consideration

## `DdEmbedding`

`DdEmbedding` embeds the number of directed donors.

## `NddEmbedding`

`NddEmbedding` embeds the number of non-directed donors.

## `NopEmbedding`

`NopEmbedding` is empty.

## `NormalizeEmbedding`

`NormalizeEmbedding` multplies every entry by a multiplier.

* `embedding : Embedding`, embedding to normalize
* `multipliers : [Float]`, entry-wise multipliers

## `OrderEmbedding`

`OrderEmbedding` embeds the order of the graph.

## `UnionEmbedding`

`UnionEmbedding` unions a set of embeddings.

## `Walk2VecEmbedding`

`Walk2VecEmbedding` embeds the graph according to a modified Walk2Vec
random walk method.

* `p0s : [DistrFun]`, initial distributions
* `tau : Nat`, steps in the random walk
* `alpha : (0, 1]`, jump probability
