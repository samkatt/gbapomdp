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
* fixes
    - [ ] think of domain spaces:
        + [ ] all should have them, and they should be 'gettable'
        + [ ] we should be able to account for continuous, maybe?
        + maybe general space:
            - model-free methods should not require discrete (obs) spaces
            - make gymspace implement it
            - make `this` implement it
* allow online learning
    - [ ] belief manager can do both 'reset' and 'new_episode' things
        + [ ] maybe additional function (input) for the belief manager
        + [ ] create 'rejection belief manager' as a belief manager
    - [ ] implement importance sampling
    - [ ] domains or environments can somehow facilitate thins
        + possibly 'reset state'
            * default is sample new state
            * implemented per domain
* generalize to other domains
    - [ ] add 'reward' and 'terminal' function in simulator
    - [ ] implement those functions in all domains
    - [ ] create a new factory function to create learned environment
    - [ ] make all domains return np.arrays as state
    - [ ] force state & observations to be from discrete
* remove all pylint things
* get ``` mypy --strict ``` working
* test https://github.ccs.neu.edu/abaisero/gym-pomdps

#### features
* use pure indices in learning dynamics (i.e. BA-POMDP)

[1]: https://papers.nips.cc/paper/8080-randomized-prior-functions-for-deep-reinforcement-learning.pdf
