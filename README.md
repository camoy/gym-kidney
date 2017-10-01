# gym-kidney

OpenAI Gym environment for kidney exchange.

## Installation

    pip install -e .

## Dependencies

* Python
* OpenAI Gym
* NumPy
* SciPy
* NetworkX

## Optional dependencies

* Matplotlib (visualization)
* OpenAI Baselines (DQN example)

## Usage

See the files in `examples/` and run them, substituting appropriate
parameters.

## Organization

The components of the environment are split into several directories.
Each component is modular and described in more detail in the `README`
located in the respective directory. Here is a brief overview.

* `_solver/` contains James Trimble's
  [kidney solver](https://github.com/jamestrimble/kidney_solver). This
  should not be modified.
* `embeddings/` contain modules which embed the kidney exchange graph
  into a fixed-size vector.
* `envs/` has only the main kidney environment driver.
* `loggers/` implement different means to record experimental output
  from the environment.
* `models/` determine how the kidney exchange evolves over time.
* `wrappers/` contains auxiliary classes for modifying the environment.

Most classes inherit from an abstract class specifying the expected
methods every subclass must implement.
