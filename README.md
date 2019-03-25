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
```pip install tensorflow ```

### Run the program
``` main.py ``` is located in ``` src ```

```console
python main.py -D cartpole -v --network_size med
python -h
```

## Development

* documentation: Run ``` ./make_documentation.sh ``` in root and find
  documentation in ``` doc/pobnrl ``` folder
* formatting: Run ``` ./check_formatting.sh ``` in root and check whether the
  code is formatted correctly
* testing: Run ``` ./run_tests.sh ``` in root and check whether all pass

### TODO
* refactoring
    - rename q_functions and the sort.. what are nets + replay buffers?
    - refactor priors away elegantly
    - create a 'batch' object
* dev
    - implement logging framework
    - implement static analysing mechanism
    - think of how to do checks (exceptions, asserts, none at all?)
* test
    - envs
    - integration (does everything *run*)

### Conventions
* use ```python FIXME ``` to identify fixes
* use ```python TODO ``` to identify TODOS
* use Pydoc style comments / docstrings

### Arbitrary decisions
* usage of huber loss (over RMSE, unless specified otherwise with --loss rmse)
* usage of Glorot normal initialization for all layers
