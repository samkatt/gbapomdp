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
``` pip install gym ```

OpenCV python:
``` pip instal opencv-python ```

ffmpeg encoding
``` sudo apt install ffmpeg ```

tensorflow
``` pip install tensorflow ```

### Run the program
```  main.py ``` is located in ``` src ```

```bash
python main.py -D cartpole -v --network_size med
python -h
```

## Relevant literature
* [Osband, Ian, John Aslanides, and Albin Cassirer. "Randomized prior functions
  for deep reinforcement learning." Advances in Neural Information Processing
      Systems. 2018.][1]

## Development

* documentation: Run ``` ./make_documentation.sh ``` in root and find
  documentation in ``` doc/pobnrl ``` folder
* formatting: Run ``` ./static_analyse.sh ``` in root and check whether the
  code is formatted correctly
* testing: Run ``` ./run_tests.sh ``` in root and check whether all pass

### TODO
* generalize to other domains
    - [ ] fix domain spaces:
        + [ ] make all domains return np.arrays as state
        + [ ] all should have all, and they should be 'gettable'
    - [ ] add 'reward' and 'terminal' function in simulator (or somewhere else?)
    - [ ] implement those functions in all domains
    - [ ] create a new factory function to create learned environment (instead of in model_based.py)
    - [ ] force state & observations to be from discrete
* generalize to other learning methods
    - [ ] generalize `reset_particle_f`
* clean up
    - [x] implement abstract iteration method for particle filters
    - [ ] make environments print each step, not at the end
* features: general space object:
    - [ ] model-free methods should not require discrete (obs) spaces
    - [ ] make gymspace implement it
    - [ ] make `this` implement it
* remove all pylint things
* get ``` mypy --strict ``` working
* test https://github.ccs.neu.edu/abaisero/gym-pomdps

#### features
* use pure indices in learning dynamics (i.e. BA-POMDP)

[1]: https://papers.nips.cc/paper/8080-randomized-prior-functions-for-deep-reinforcement-learning.pdf
