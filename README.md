# Partially Observable Neural Reinforcement Learning

A code base to run (Model-based Bayesian Neural) Reinforcement Learning
experiments on partially observable domains. This project is meant for
reinforcement learning researchers to compare different methods. It contains
various different environments to test the methods on. Note that this project
has mostly been written for personal use, research, and thus may lack the
documentation that one would typically expect from open source projects.

## Use

### Installation
Install the required python packages and dependencies

#### Known dependencies

Open AI gym:

```
pip install gym
```

OpenCV python:
```
pip instal opencv-python
```

ffmpeg encoding
```
sudo apt install ffmpeg
```

tensorflow
```
pip install tensorflow
```

### Run the program
`main.py` is located in `src`

```bash
python main.py -D cartpole -v --network_size med
python -h
```

## Relevant literature
* [Osband, Ian, John Aslanides, and Albin Cassirer. "Randomized prior functions
for deep reinforcement learning." Advances in Neural Information Processing
Systems. 2018.][1]

## Development

* documentation: Run `./make_documentation.sh` in root and find documentation
  in `doc/pobnrl` folder
* static analysis & formatting: Run `./static_analyse.sh` in root and check
  whether the code is formatted correctly
* testing: Run `./run_tests.sh` in root and check whether all pass

### TODO

* get `mypy --strict` working
* test `https://github.ccs.neu.edu/abaisero/gym-pomdps`
* move to keras
* profiling (add numba @jit to functions where applicable for speed ups)

#### spaces issue

Observations:
1. Planning requires (small) discrete action & observation problems
2. Current implementation of learned environments assumes small discrete state space
3. Model-free methods have been using one-hot, while learned model is much better at the opposite

Questions:
* Continuous versus discrete
    1. Should I go to continuous states (learned environments)?
    2. Should I go to continuous planning things?
* Should I make a more general space, and work with gym spaces?
* How do I go about different representations of the observations, states and actions?

#### features
* use pure indices in learning dynamics (i.e. BA-POMDP)

[1]: https://papers.nips.cc/paper/8080-randomized-prior-functions-for-deep-reinforcement-learning.pdf
