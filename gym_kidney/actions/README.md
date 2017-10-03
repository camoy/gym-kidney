# Actions

Actions define what any agent can do in the environment.

## `FlapAction`

`FlapAction` performs a maximum cardinality matching on
the graph.

* `cycle_cap : Nat`, the cycle cap for the solver
* `chain_cap : Nat`, the chain cap for the solver

## `BloodAction`

`BloodAction` reweights the graph edges according to the
the action before calling the solver.

* `cycle_cap : Nat`, the cycle cap for the solver
* `chain_cap : Nat`, the chain cap for the solver
* `min : Real`, smallest value for vertex
* `max : Real`, largest value for vertex
* `w_fun : (Real, Real -> Real)`, weight function
