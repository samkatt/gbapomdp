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
* formatting: Run ``` ./check_formatting.sh ``` in root and check whether the
  code is formatted correctly
* testing: Run ``` ./run_tests.sh ``` in root and check whether all pass

### TODO
* implement model-based agent:
    - implement PO-UCT
    - implement belief updates
    - implement POMCP agent
        + test on environments
    - implement NNs as POMDP models
    - implement simulator
* FIX
    - figure out why num_nets 2 takes *so much longer*
* refactoring
    - variables with _ or not, then always properties, what's the deal?
        + conf without _, rest with
        + property for non-confs that need to be accessible
    - rename and reinvent q_functions
    - refactor priors away elegantly
        + [maybe] refactor the whole creation of training operation..
        + improved general network and losses type of approach
    - create a 'batch' class and use that instead of individual placeholders

[1]: https://papers.nips.cc/paper/8080-randomized-prior-functions-for-deep-reinforcement-learning.pdf
