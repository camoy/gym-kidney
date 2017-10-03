# Models

Models describe how the graph evolves over time.

## `HomogeneousModel`

HomogeneousModel evolves the graph according to a homogeneous
Erdős–Rényi random model.

* `m : Nat`, expected vertices per period
* `k : Nat`, ticks per period
* `p : [0, 1]`, probability of edge between vertices
* `p_a : [0, 1]`, probability of NDD
* `len : Nat`, ticks per episode

## `HeterogeneousModel`

HeterogeneousModel evolves the graph according to a heterogeneous
Erdős–Rényi random model.

* `m : Nat`, expected vertices per period
* `k : Nat`, ticks per period
* `p_l : [0, 1]`, probability of edge to patient with low PRA
* `p_h : [0, 1]`, probability of edge to patient with high PRA
* `p_a : [0, 1]`, probability of NDD
* `p_s : [0, 1]`, probability of patient with high PRA
* `len : Nat`, ticks per episode

## `DataModel`

DataModel evolves the graph based on actual kidney exchange data.

* `m : Nat`, expected vertices per period
* `k : Nat`, ticks per period
* `data : String`, path to CSV containing data
* `details : String`, path to CSV containing vertex attributes
* `len : Nat`, ticks per episode
